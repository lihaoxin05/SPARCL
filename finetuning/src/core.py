import datetime
import os
import random
import copy
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from PIL import Image

from src.adain.utils import AdaIN
from src.datasets import dataset_types
from src.lib.CLIP.clip import clip as clip
from src.lib.CLIP.clip.model import convert_weights
from src.random_augmenter import random_augment
from src.utils.augmentations import RandomAugmentation
from src.lib.CLIP.clip.simple_tokenizer import SimpleTokenizer as _Tokenizer

from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group


_tokenizer = _Tokenizer()


def ddp_setup(rank: int, world_size: int, master_port: str):
  """
  Args:
      rank: Unique identifier of each process
     world_size: Total number of processes
  """
  os.environ["MASTER_ADDR"] = "localhost"
  os.environ["MASTER_PORT"] = master_port
  init_process_group(backend="nccl", rank=rank, world_size=world_size)
  torch.cuda.set_device(rank)


def ada_margin_loss(logits_per_image, logits_per_text, threshold, image_group_number, text_group_number, additional_real_weight=0.0, margin_low=-2.0, margin_scale=1.0):
    b1, b2 = logits_per_image.shape
    assert b1 // image_group_number * text_group_number == b2
    b1, b2 = logits_per_text.shape
    assert b1 // text_group_number * image_group_number == b2
    assert image_group_number == 3 and text_group_number == 3, "Not Implemented."
    
    total_b = logits_per_image.shape[0]
    batch_b = total_b // image_group_number
    positions_0 = torch.arange(total_b)
    ori_positions_0 = positions_0[:batch_b]
    neg_positions_0 = positions_0[batch_b:2*batch_b]
    pos_positions_0 = positions_0[2*batch_b:]

    ### ori
    mask = torch.ones_like(logits_per_image, dtype=torch.bool)
    mask[neg_positions_0,:] = 0
    mask[pos_positions_0,:] = 0
    mask[ori_positions_0, ori_positions_0] = 0
    mask[ori_positions_0, neg_positions_0] = 0
    mask[ori_positions_0, pos_positions_0] = 0

    ## image
    # ori ~ neg
    comp = (logits_per_image[ori_positions_0, ori_positions_0] - logits_per_image[ori_positions_0, neg_positions_0]).detach()
    ada_threshold_1 = torch.ones_like(comp) * threshold
    ada_threshold_1 = torch.where(comp < margin_low, comp, ada_threshold_1)
    ada_threshold_1 = torch.where((comp >= margin_low)&(comp <= threshold), ((1-(comp-margin_low)/(threshold-margin_low))*margin_scale+1)*threshold, ada_threshold_1)

    # pos ~ neg
    comp = (logits_per_image[ori_positions_0, pos_positions_0] - logits_per_image[ori_positions_0, neg_positions_0]).detach()
    ada_threshold_2 = torch.ones_like(comp) * threshold
    ada_threshold_2 = torch.where(comp<margin_low, comp, ada_threshold_2)
    ada_threshold_2 = torch.where((comp>=margin_low)&(comp<=threshold), ((1-(comp-margin_low)/(threshold-margin_low))*margin_scale+1)*threshold, ada_threshold_2)

    # neg ~ other
    comp = (logits_per_image[ori_positions_0, neg_positions_0] - torch.max(logits_per_image[mask].view(batch_b, -1), dim=-1)[0]).detach()
    ada_threshold_3 = torch.ones_like(comp) * threshold
    ada_threshold_3 = torch.where(comp<margin_low, comp, ada_threshold_3)
    ada_threshold_3 = torch.where((comp>=margin_low)&(comp<=threshold), ((1-(comp-margin_low)/(threshold-margin_low))*margin_scale+1)*threshold, ada_threshold_3)

    ## text
    # ori ~ neg
    comp = (logits_per_text[ori_positions_0, ori_positions_0] - logits_per_text[ori_positions_0, neg_positions_0]).detach()
    ada_threshold_4 = torch.ones_like(comp) * threshold
    ada_threshold_4 = torch.where(comp<margin_low, comp, ada_threshold_4)
    ada_threshold_4 = torch.where((comp>=margin_low)&(comp<=threshold), ((1-(comp-margin_low)/(threshold-margin_low))*margin_scale+1)*threshold, ada_threshold_4)

    # pos ~ neg
    comp = (logits_per_text[ori_positions_0, pos_positions_0] - logits_per_text[ori_positions_0, neg_positions_0]).detach()
    ada_threshold_5 = torch.ones_like(comp) * threshold
    ada_threshold_5 = torch.where(comp<margin_low, comp, ada_threshold_5)
    ada_threshold_5 = torch.where((comp>=margin_low)&(comp<=threshold), ((1-(comp-margin_low)/(threshold-margin_low))*margin_scale+1)*threshold, ada_threshold_5)

    # neg ~ other
    comp = (logits_per_text[ori_positions_0, neg_positions_0] - torch.max(logits_per_text[mask].view(batch_b, -1), dim=-1)[0]).detach()
    ada_threshold_6 = torch.ones_like(comp) * threshold
    ada_threshold_6 = torch.where(comp<margin_low, comp, ada_threshold_6)
    ada_threshold_6 = torch.where((comp>=margin_low)&(comp<=threshold), ((1-(comp-margin_low)/(threshold-margin_low))*margin_scale+1)*threshold, ada_threshold_6)

    # loss
    total_loss = nn.functional.relu(ada_threshold_1 + logits_per_image[ori_positions_0, neg_positions_0] - logits_per_image[ori_positions_0, ori_positions_0]).mean() + \
    nn.functional.relu(ada_threshold_2 + logits_per_image[ori_positions_0, neg_positions_0] - logits_per_image[ori_positions_0, pos_positions_0]).mean() + \
    nn.functional.relu(ada_threshold_3.unsqueeze(1) + logits_per_image[mask].view(batch_b, -1) - logits_per_image[ori_positions_0, neg_positions_0].unsqueeze(1)).mean() + \
    nn.functional.relu(ada_threshold_4 + logits_per_text[ori_positions_0, neg_positions_0] - logits_per_text[ori_positions_0, ori_positions_0]).mean() + \
    nn.functional.relu(ada_threshold_5 + logits_per_text[ori_positions_0, neg_positions_0] - logits_per_text[ori_positions_0, pos_positions_0]).mean() + \
    nn.functional.relu(ada_threshold_6.unsqueeze(1) + logits_per_text[mask].view(batch_b, -1) - logits_per_text[ori_positions_0, neg_positions_0].unsqueeze(1)).mean()

    if additional_real_weight > 0.0:
        comp = (logits_per_image[ori_positions_0, ori_positions_0] - torch.max(logits_per_image[mask].view(batch_b, -1)[:,:(batch_b-1)], dim=-1)[0]).detach()
        ada_threshold_1 = torch.ones_like(comp) * threshold
        ada_threshold_1 = torch.where(comp<margin_low, comp, ada_threshold_1)
        ada_threshold_1 = torch.where((comp>=margin_low)&(comp<=threshold), ((1-(comp-margin_low)/(threshold-margin_low))*margin_scale+1)*threshold, ada_threshold_1)

        comp = (logits_per_text[ori_positions_0, pos_positions_0] - torch.max(logits_per_text[mask].view(batch_b, -1)[:,:(batch_b-1)], dim=-1)[0]).detach()
        ada_threshold_2 = torch.ones_like(comp) * threshold
        ada_threshold_2 = torch.where(comp<margin_low, comp, ada_threshold_2)
        ada_threshold_2 = torch.where((comp>=margin_low)&(comp<=threshold), ((1-(comp-margin_low)/(threshold-margin_low))*margin_scale+1)*threshold, ada_threshold_2)

        total_loss += additional_real_weight * (nn.functional.relu(ada_threshold_1.unsqueeze(1) + logits_per_image[mask].view(batch_b, -1)[:,:(batch_b-1)] - logits_per_image[ori_positions_0, ori_positions_0].unsqueeze(1)).mean() + \
        nn.functional.relu(ada_threshold_2.unsqueeze(1) + logits_per_text[mask].view(batch_b, -1)[:,:(batch_b-1)] - logits_per_text[ori_positions_0, ori_positions_0].unsqueeze(1)).mean())
    
    ### neg
    mask = torch.ones_like(logits_per_image, dtype=torch.bool)
    mask[ori_positions_0,:] = 0
    mask[pos_positions_0,:] = 0
    mask[neg_positions_0, ori_positions_0] = 0
    mask[neg_positions_0, neg_positions_0] = 0
    mask[neg_positions_0, pos_positions_0] = 0

    ## image
    # neg ~ ori
    comp = (logits_per_image[neg_positions_0, neg_positions_0] - logits_per_image[neg_positions_0, ori_positions_0]).detach()
    ada_threshold_1 = torch.ones_like(comp) * threshold
    ada_threshold_1 = torch.where(comp<margin_low, comp, ada_threshold_1)
    ada_threshold_1 = torch.where((comp>=margin_low)&(comp<=threshold), ((1-(comp-margin_low)/(threshold-margin_low))*margin_scale+1)*threshold, ada_threshold_1)

    # neg ~ pos
    comp = (logits_per_image[neg_positions_0, neg_positions_0] - logits_per_image[neg_positions_0, pos_positions_0]).detach()
    ada_threshold_2 = torch.ones_like(comp) * threshold
    ada_threshold_2 = torch.where(comp<margin_low, comp, ada_threshold_2)
    ada_threshold_2 = torch.where((comp>=margin_low)&(comp<=threshold), ((1-(comp-margin_low)/(threshold-margin_low))*margin_scale+1)*threshold, ada_threshold_2)

    # ori ~ other
    comp = (logits_per_image[neg_positions_0, ori_positions_0] - torch.max(logits_per_image[mask].view(batch_b, -1), dim=-1)[0]).detach()
    ada_threshold_3 = torch.ones_like(comp) * threshold
    ada_threshold_3 = torch.where(comp<margin_low, comp, ada_threshold_3)
    ada_threshold_3 = torch.where((comp>=margin_low)&(comp<=threshold), ((1-(comp-margin_low)/(threshold-margin_low))*margin_scale+1)*threshold, ada_threshold_3)

    # pos ~ other
    comp = (logits_per_image[neg_positions_0, pos_positions_0] - torch.max(logits_per_image[mask].view(batch_b, -1), dim=-1)[0]).detach()
    ada_threshold_4 = torch.ones_like(comp) * threshold
    ada_threshold_4 = torch.where(comp<margin_low, comp, ada_threshold_4)
    ada_threshold_4 = torch.where((comp>=margin_low)&(comp<=threshold), ((1-(comp-margin_low)/(threshold-margin_low))*margin_scale+1)*threshold, ada_threshold_4)

    ## text
    # neg ~ ori
    comp = (logits_per_text[neg_positions_0, neg_positions_0] - logits_per_text[neg_positions_0, ori_positions_0]).detach()
    ada_threshold_5 = torch.ones_like(comp) * threshold
    ada_threshold_5 = torch.where(comp<margin_low, comp, ada_threshold_5)
    ada_threshold_5 = torch.where((comp>=margin_low)&(comp<=threshold), ((1-(comp-margin_low)/(threshold-margin_low))*margin_scale+1)*threshold, ada_threshold_5)

    # neg ~ pos
    comp = (logits_per_text[neg_positions_0, neg_positions_0] - logits_per_text[neg_positions_0, pos_positions_0]).detach()
    ada_threshold_6 = torch.ones_like(comp) * threshold
    ada_threshold_6 = torch.where(comp<margin_low, comp, ada_threshold_6)
    ada_threshold_6 = torch.where((comp>=margin_low)&(comp<=threshold), ((1-(comp-margin_low)/(threshold-margin_low))*margin_scale+1)*threshold, ada_threshold_6)

    # ori ~ other
    comp = (logits_per_text[neg_positions_0, ori_positions_0] - torch.max(logits_per_text[mask].view(batch_b, -1), dim=-1)[0]).detach()
    ada_threshold_7 = torch.ones_like(comp) * threshold
    ada_threshold_7 = torch.where(comp<margin_low, comp, ada_threshold_7)
    ada_threshold_7 = torch.where((comp>=margin_low)&(comp<=threshold), ((1-(comp-margin_low)/(threshold-margin_low))*margin_scale+1)*threshold, ada_threshold_7)

    # pos ~ other
    comp = (logits_per_text[neg_positions_0, pos_positions_0] - torch.max(logits_per_text[mask].view(batch_b, -1), dim=-1)[0]).detach()
    ada_threshold_8 = torch.ones_like(comp) * threshold
    ada_threshold_8 = torch.where(comp<margin_low, comp, ada_threshold_8)
    ada_threshold_8 = torch.where((comp>=margin_low)&(comp<=threshold), ((1-(comp-margin_low)/(threshold-margin_low))*margin_scale+1)*threshold, ada_threshold_8)

    # loss
    total_loss += nn.functional.relu(ada_threshold_1 + logits_per_image[neg_positions_0, ori_positions_0] - logits_per_image[neg_positions_0, neg_positions_0]).mean() + \
    nn.functional.relu(ada_threshold_2 + logits_per_image[neg_positions_0, pos_positions_0] - logits_per_image[neg_positions_0, neg_positions_0]).mean() + \
    nn.functional.relu(ada_threshold_3.unsqueeze(1) + logits_per_image[mask].view(batch_b, -1) - logits_per_image[neg_positions_0, ori_positions_0].unsqueeze(1)).mean() + \
    nn.functional.relu(ada_threshold_4.unsqueeze(1) + logits_per_image[mask].view(batch_b, -1) - logits_per_image[neg_positions_0, pos_positions_0].unsqueeze(1)).mean() + \
    nn.functional.relu(ada_threshold_5 + logits_per_text[neg_positions_0, ori_positions_0] - logits_per_text[neg_positions_0, neg_positions_0]).mean() + \
    nn.functional.relu(ada_threshold_6 + logits_per_text[neg_positions_0, pos_positions_0] - logits_per_text[neg_positions_0, neg_positions_0]).mean() + \
    nn.functional.relu(ada_threshold_7.unsqueeze(1) + logits_per_text[mask].view(batch_b, -1) - logits_per_text[neg_positions_0, ori_positions_0].unsqueeze(1)).mean() + \
    nn.functional.relu(ada_threshold_8.unsqueeze(1) + logits_per_text[mask].view(batch_b, -1) - logits_per_text[neg_positions_0, pos_positions_0].unsqueeze(1)).mean()

    if additional_real_weight > 0.0:
        comp = (logits_per_image[neg_positions_0, neg_positions_0] - torch.max(logits_per_image[mask].view(batch_b, -1)[:,:(batch_b-1)], dim=-1)[0]).detach()
        ada_threshold_1 = torch.ones_like(comp) * threshold
        ada_threshold_1 = torch.where(comp<margin_low, comp, ada_threshold_1)
        ada_threshold_1 = torch.where((comp>=margin_low)&(comp<=threshold), ((1-(comp-margin_low)/(threshold-margin_low))*margin_scale+1)*threshold, ada_threshold_1)

        comp = (logits_per_text[neg_positions_0, neg_positions_0] - torch.max(logits_per_text[mask].view(batch_b, -1)[:,:(batch_b-1)], dim=-1)[0]).detach()
        ada_threshold_2 = torch.ones_like(comp) * threshold
        ada_threshold_2 = torch.where(comp<margin_low, comp, ada_threshold_2)
        ada_threshold_2 = torch.where((comp>=margin_low)&(comp<=threshold), ((1-(comp-margin_low)/(threshold-margin_low))*margin_scale+1)*threshold, ada_threshold_2)

        total_loss += additional_real_weight * (nn.functional.relu(ada_threshold_1.unsqueeze(1) + logits_per_image[mask].view(batch_b, -1)[:,:(batch_b-1)] - logits_per_image[neg_positions_0, neg_positions_0].unsqueeze(1)).mean() + nn.functional.relu(ada_threshold_2.unsqueeze(1) + logits_per_text[mask].view(batch_b, -1)[:,:(batch_b-1)] - logits_per_text[neg_positions_0, neg_positions_0].unsqueeze(1)).mean())

    ### pos
    mask = torch.ones_like(logits_per_image, dtype=torch.bool)
    mask[ori_positions_0,:] = 0
    mask[neg_positions_0,:] = 0
    mask[pos_positions_0, ori_positions_0] = 0
    mask[pos_positions_0, neg_positions_0] = 0
    mask[pos_positions_0, pos_positions_0] = 0

    ## image
    # ori ~ neg
    comp = (logits_per_image[pos_positions_0, ori_positions_0] - logits_per_image[pos_positions_0, neg_positions_0]).detach()
    ada_threshold_1 = torch.ones_like(comp) * threshold
    ada_threshold_1 = torch.where(comp<margin_low, comp, ada_threshold_1)
    ada_threshold_1 = torch.where((comp>=margin_low)&(comp<=threshold), ((1-(comp-margin_low)/(threshold-margin_low))*margin_scale+1)*threshold, ada_threshold_1)

    # pos ~ neg
    comp = (logits_per_image[pos_positions_0, pos_positions_0] - logits_per_image[pos_positions_0, neg_positions_0]).detach()
    ada_threshold_2 = torch.ones_like(comp) * threshold
    ada_threshold_2 = torch.where(comp<margin_low, comp, ada_threshold_2)
    ada_threshold_2 = torch.where((comp>=margin_low)&(comp<=threshold), ((1-(comp-margin_low)/(threshold-margin_low))*margin_scale+1)*threshold, ada_threshold_2)

    # neg ~ other
    comp = (logits_per_image[pos_positions_0, neg_positions_0] - torch.max(logits_per_image[mask].view(batch_b, -1), dim=-1)[0]).detach()
    ada_threshold_3 = torch.ones_like(comp) * threshold
    ada_threshold_3 = torch.where(comp<margin_low, comp, ada_threshold_3)
    ada_threshold_3 = torch.where((comp>=margin_low)&(comp<=threshold), ((1-(comp-margin_low)/(threshold-margin_low))*margin_scale+1)*threshold, ada_threshold_3)

    ## text
    # ori ~ neg
    comp = (logits_per_text[pos_positions_0, ori_positions_0] - logits_per_text[pos_positions_0, neg_positions_0]).detach()
    ada_threshold_4 = torch.ones_like(comp) * threshold
    ada_threshold_4 = torch.where(comp<margin_low, comp, ada_threshold_4)
    ada_threshold_4 = torch.where((comp>=margin_low)&(comp<=threshold), ((1-(comp-margin_low)/(threshold-margin_low))*margin_scale+1)*threshold, ada_threshold_4)

    # pos ~ neg
    comp = (logits_per_text[pos_positions_0, pos_positions_0] - logits_per_text[pos_positions_0, neg_positions_0]).detach()
    ada_threshold_5 = torch.ones_like(comp) * threshold
    ada_threshold_5 = torch.where(comp<margin_low, comp, ada_threshold_5)
    ada_threshold_5 = torch.where((comp>=margin_low)&(comp<=threshold), ((1-(comp-margin_low)/(threshold-margin_low))*margin_scale+1)*threshold, ada_threshold_5)

    # neg ~ other
    comp = (logits_per_text[pos_positions_0, neg_positions_0] - torch.max(logits_per_text[mask].view(batch_b, -1), dim=-1)[0]).detach()
    ada_threshold_6 = torch.ones_like(comp) * threshold
    ada_threshold_6 = torch.where(comp<margin_low, comp, ada_threshold_6)
    ada_threshold_6 = torch.where((comp>=margin_low)&(comp<=threshold), ((1-(comp-margin_low)/(threshold-margin_low))*margin_scale+1)*threshold, ada_threshold_6)

    # loss
    total_loss += nn.functional.relu(ada_threshold_1 + logits_per_image[pos_positions_0, neg_positions_0] - logits_per_image[pos_positions_0, ori_positions_0]).mean() + \
    nn.functional.relu(ada_threshold_2 + logits_per_image[pos_positions_0, neg_positions_0] - logits_per_image[pos_positions_0, pos_positions_0]).mean() + \
    nn.functional.relu(ada_threshold_3.unsqueeze(1) + logits_per_image[mask].view(batch_b, -1) - logits_per_image[pos_positions_0, neg_positions_0].unsqueeze(1)).mean() + \
    nn.functional.relu(ada_threshold_4 + logits_per_text[pos_positions_0, neg_positions_0] - logits_per_text[pos_positions_0, ori_positions_0]).mean() + \
    nn.functional.relu(ada_threshold_5 + logits_per_text[pos_positions_0, neg_positions_0] - logits_per_text[pos_positions_0, pos_positions_0]).mean() + \
    nn.functional.relu(ada_threshold_6.unsqueeze(1) + logits_per_text[mask].view(batch_b, -1) - logits_per_text[pos_positions_0, neg_positions_0].unsqueeze(1)).mean()

    if additional_real_weight > 0.0:
        comp = (logits_per_image[pos_positions_0, pos_positions_0] - torch.max(logits_per_image[mask].view(batch_b, -1)[:,:(batch_b-1)], dim=-1)[0]).detach()
        ada_threshold_1 = torch.ones_like(comp) * threshold
        ada_threshold_1 = torch.where(comp<margin_low, comp, ada_threshold_1)
        ada_threshold_1 = torch.where((comp>=margin_low)&(comp<=threshold), ((1-(comp-margin_low)/(threshold-margin_low))*margin_scale+1)*threshold, ada_threshold_1)

        comp = (logits_per_text[pos_positions_0, pos_positions_0] - torch.max(logits_per_text[mask].view(batch_b, -1)[:,:(batch_b-1)], dim=-1)[0]).detach()
        ada_threshold_2 = torch.ones_like(comp) * threshold
        ada_threshold_2 = torch.where(comp<margin_low, comp, ada_threshold_2)
        ada_threshold_2 = torch.where((comp>=margin_low)&(comp<=threshold), ((1-(comp-margin_low)/(threshold-margin_low))*margin_scale+1)*threshold, ada_threshold_2)

        total_loss += additional_real_weight * (nn.functional.relu(ada_threshold_1.unsqueeze(1) + logits_per_image[mask].view(batch_b, -1)[:,:(batch_b-1)] - logits_per_image[pos_positions_0, pos_positions_0].unsqueeze(1)).mean() + nn.functional.relu(ada_threshold_2.unsqueeze(1) + logits_per_text[mask].view(batch_b, -1)[:,:(batch_b-1)] - logits_per_text[pos_positions_0, pos_positions_0].unsqueeze(1)).mean())

    return total_loss


class FT_CLIP:
    """
    Class for fine-tuning CLIP with specified configurations.
    """

    def __init__(self, local_rank, args):
        """
        Initialize the FT_CLIP object with provided arguments.
        :param args: Object containing training parameters.
        """
        self.args = args
        self.args.local_rank = local_rank

        # Set lr based on batch size, set group number
        self.args.lr = self.args.base_lr * self.args.batch_size * self.args.world_size / 512
        # self.args.lr = self.args.base_lr
        if "COCO_neg_pos" in self.args.data_type:
            if not self.args.random_select_cols:
                self.image_group_number = len(self.args.data_path)
                self.text_group_number = len(self.args.ori_text_cols) + len(self.args.annotation_path) * (len(self.args.negative_text_cols) + len(self.args.positive_text_cols))
            else:
                if len(self.args.data_path) > 1:
                    if len(self.args.positive_text_cols) > 0: # real + negative + positive
                        self.image_group_number = 1 + len(self.args.annotation_path) * 2
                    else: # only real images and negative images
                        self.image_group_number = 1 + len(self.args.annotation_path) * 1
                else:
                    self.image_group_number = 1 # only real images
                if len(self.args.positive_text_cols) > 0:
                    self.text_group_number = 1 + len(self.args.annotation_path) * 2
                else:
                    self.text_group_number = 1 + len(self.args.annotation_path) * 1
        else:
            assert False, "Not Implemented."
        if self.args.local_rank == 0:
            print("- Base learning rate:", self.args.base_lr, "; learning rate:", self.args.lr)

        ddp_setup(self.args.local_rank, self.args.world_size, self.args.master_port)

        # Set random seed
        if self.args.local_rank == 0:
            print("- Set seed:", args.seed)
        self.seed_everything(args.seed)

        # Load model
        if self.args.local_rank == 0:
            print("- Load model")
        self.load_model(base_name=args.backbone_name, weight_name=args.load_from_path)

        # Setting to training mode
        if self.args.local_rank == 0:
            print("- Set train model")
        self.set_train_state()

        # Load style module
        self.load_style_module()

        # Load train dataset
        if self.args.local_rank == 0:
            print(f"- Load dataset - {args.data_type} ")
        self.load_train_dataset(args.data_type, args.heavy_aug)

        # Train the model
        if self.args.local_rank == 0:
            print("- Start training")
        self.train_model()

        # Done!
        if self.args.local_rank == 0:
            print("Done...")

    def seed_everything(self, seed: int):
        """
        Set a random seed for reproducibility.
        :param seed: Integer seed value.
        """
        random.seed(seed)
        os.environ["PYTHONHASHSEED"] = str(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = True

    def load_model(self, base_name="RN50", weight_name=""):
        """
        Load the CLIP model and display its parameters.

        :param base_name: Base model name.
        :param weight_name: Pre-trained weight path.
        """

        self.model, self.preprocess = clip.load(
            base_name,
            jit=False,
            lora=self.args.lora_r,
            mixstyle=self.args.mixstyle,
            all_features=self.args.all_features,
            sigmoid=self.args.sigmoid,
            sigmoid_logit_init=self.args.sigmoid_logit_init,
            trainable_logit=self.args.trainable_logit,
        )
        if self.args.lora_r <= 0:
            convert_weights(self.model)

        if weight_name:
            loaded_model = torch.load(weight_name)
            self.model.load_state_dict(loaded_model["model"])
            self.start_epoch = loaded_model["epoch"] + 1
        else:
            self.start_epoch = 0

        if self.args.local_rank == 0:
            print("=========")
            print(
                "Model parameters:",
                f"{np.sum([int(np.prod(p.shape)) for p in self.model.parameters()]):,}",
            )
            print("Input resolution:", self.model.visual.input_resolution)
            print("Context length:", self.model.context_length)
            print("Vocab size:", self.model.vocab_size)
            print("=========")
        
        self.model = DDP(self.model, find_unused_parameters=True)


    def load_style_module(self):
        """
        Load style module
        """
        if self.args.style_transfer == 'adain':
            self.style_module = AdaIN()
        else:
            self.style_module = None

    def set_train_state(self):
        self.model.train()

    def set_eval_state(self):
        self.model.eval()

    def convert_models_to_fp32(self, model):
        for p in model.parameters():
            if p.requires_grad:
                p.data = p.data.float()
                p.grad.data = p.grad.data.float()

    def load_train_dataset(self, data_type, heavy_aug):
        """
        Load training dataset with optional augmentations.

        :param data_type: Type of data to be loaded.
        :param heavy_aug: Boolean flag to use heavy augmentations.
        """
        if heavy_aug:
            self.preprocess.transforms.insert(
                0, RandomAugmentation(random_augment())
            )  # adding random_augmentations

        if "COCO_neg_pos" in data_type:
            self.dataset = dataset_types["COCO_neg_pos"](
                annotation_path=self.args.annotation_path,
                image_path=self.args.data_path,
                ori_text_cols=self.args.ori_text_cols,
                negative_text_cols=self.args.negative_text_cols,
                positive_text_cols=self.args.positive_text_cols,
                random_select_cols=self.args.random_select_cols,
                transform=self.preprocess,
            )
        else:
            assert False, "Not Implemented."

        self.dataset_loader = torch.utils.data.DataLoader(
            self.dataset,
            batch_size=self.args.batch_size,
            num_workers=self.args.workers,
            shuffle=False, # shuffle in sampler
            sampler=DistributedSampler(self.dataset, shuffle=True),
            drop_last=True,
        )

    def construct_save_path(self, starting_time, extension, suffix=""):
        trained = "ALL"
        arch_name = self.args.backbone_name.replace("/", "")
        folder_name = self.args.folder_name
        if not folder_name:
            folder_name = arch_name

        components = [
            f"CLIP_{arch_name}_finetune",
            f"{trained}",
            f"LoraRank{self.args.lora_r}",
            f"LR{self.args.base_lr:.8f}",
            f"WD{self.args.weight_decay:.4f}",
            f"epochs{self.args.epochs}",
            f"steps{self.args.stop_steps}",
            f"seed{self.args.seed}",
        ]
        if suffix:
            components.append(suffix)

        fname = "_".join(components) + f".{extension}"
        fpath = f"{folder_name}/{'+'.join(self.args.data_type)}_{self.args.tag}"
        os.makedirs(fpath, exist_ok=True)

        fpath = f"{fpath}/{fname}"
        return fpath.replace(" ", "_").replace(":", "_")

    def train_model(self):
        epochs = self.args.epochs

        if self.args.local_rank == 0:
            starting_time = datetime.datetime.now()
            print("== Training started at:", starting_time)
            print("Dataloader length:", len(self.dataset_loader))

        all_losses = []
        all_lrs = []

        ### optimizer and scheduler
        trainable_parameters = [p for p in self.model.parameters() if p.requires_grad]
        model_optimizer = torch.optim.AdamW(
            trainable_parameters,
            lr=self.args.lr,
            betas=(self.args.beta1, self.args.beta2),
            eps=self.args.eps,
            weight_decay=self.args.weight_decay,
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(model_optimizer, min(epochs*len(self.dataset_loader), self.args.stop_steps)) # scheduler as steps
        ###

        ### loss
        if not self.args.sigmoid:
            loss_img = nn.CrossEntropyLoss(reduce=not (self.args.group and self.args.selecting_alg[0] >= 0))
            loss_txt = nn.CrossEntropyLoss(reduce=not (self.args.group and self.args.selecting_alg[0] >= 0))
        else:
            loss_img = nn.BCEWithLogitsLoss()
            loss_txt = nn.BCEWithLogitsLoss()
        ###
        
        ### set start epoch and global step
        if self.start_epoch > 0:
            for _ in range(self.start_epoch*len(self.dataset_loader)):
                scheduler.step()
        global_step = self.start_epoch * len(self.dataset_loader)
        ###

        for epoch in range(self.start_epoch, epochs):
            self.dataset_loader.sampler.set_epoch(epoch)

            ### loss logger initialization
            if self.args.local_rank == 0:
                log_total_loss = 0
                log_loss_cls = 0
            ###
            if self.args.local_rank == 0:
                pbar = tqdm(enumerate(self.dataset_loader), total=len(self.dataset_loader))
            else:
                pbar = enumerate(self.dataset_loader)
            for i, data in pbar:
                model_optimizer.zero_grad()
                
                if "COCO_neg_pos" in self.args.data_type:
                    assert len(data[0]) == self.image_group_number
                    assert len(data[1]) == self.text_group_number
                    images = torch.concat(data[0], dim=0)
                    descriptions_per_sample = sum(data[1],())
                ### style transfer for images
                if self.args.style_transfer == 'adain' and self.style_module is not None:
                    with torch.no_grad():
                        images = images.cuda()
                        num_img_per_group = images.shape[0] // self.image_group_number
                        # only transfer synthetic images
                        trg_img = torch.cat([images[:num_img_per_group]] * (self.image_group_number - 1), dim=0)
                        src_img = images[num_img_per_group:]
                        clip_mean = torch.Tensor((0.48145466, 0.4578275, 0.40821073)).to(images.device)
                        clip_std = torch.Tensor((0.26862954, 0.26130258, 0.27577711)).to(images.device)
                        trg_img = trg_img * clip_std.unsqueeze(-1).unsqueeze(-1).unsqueeze(0) + clip_mean.unsqueeze(-1).unsqueeze(-1).unsqueeze(0)
                        src_img = src_img * clip_std.unsqueeze(-1).unsqueeze(-1).unsqueeze(0) + clip_mean.unsqueeze(-1).unsqueeze(-1).unsqueeze(0)
                        src_img = self.style_module.transfer(src_img, trg_img, preserve_color=False, alpha=self.args.style_transfer_alpha)
                        src_img = (src_img - torch.min(src_img)) / (torch.max(src_img) - torch.min(src_img))
                        images[num_img_per_group:] = (src_img - clip_mean.unsqueeze(-1).unsqueeze(-1).unsqueeze(0)) / clip_std.unsqueeze(-1).unsqueeze(-1).unsqueeze(0)
                ###
                ### split caption
                if self.args.capsplit:
                    split_descriptions_per_sample = []
                    for d in descriptions_per_sample:
                        split_d = [di.strip() for di in d.split(".") if di.strip()]
                        split_d = [
                            ". ".join(split_d[di : di + 4]).strip()
                            for di in range(0, len(split_d), 4)
                        ]
                        split_descriptions_per_sample.append(split_d[:1])
                    descriptions_per_sample = split_descriptions_per_sample
                ###
                ### to device
                images = images.cuda()
                descriptions_tokens_atts = []
                for di in descriptions_per_sample:
                    d_tokens = clip.tokenize(di, truncate=True).cuda()
                    descriptions_tokens_atts.append(d_tokens)
                descriptions_tokens_atts = torch.nn.utils.rnn.pad_sequence(
                    descriptions_tokens_atts, batch_first=True
                )
                ###
                ### all features
                if self.args.all_features:
                    logits_per_image, logits_per_text, image_features, text_features, image_tokens, text_tokens = self.model(
                        images, descriptions_tokens_atts
                    )
                    # shape:
                    # logits_per_image: b*img_group, b*text_group
                    # logits_per_text: b*text_group, b*img_group
                    # image_features: b*img_group, d
                    # text_features: b*text_group, d
                    # image_tokens: b*img_group, patch+1, d
                    # text_tokens: b*text_group, 77, d
                else:
                    logits_per_image, logits_per_text = self.model(
                        images, descriptions_tokens_atts
                    )
                ###
                ### construct ground truth
                if "COCO_neg_pos" in self.args.data_type:
                    num_sample_per_group = logits_per_image.shape[0] // self.image_group_number
                    assert num_sample_per_group == logits_per_text.shape[0] // self.text_group_number
                    ground_truth_ind = torch.arange(num_sample_per_group, dtype=torch.long).cuda()
                    ground_truth_one_hot = torch.nn.functional.one_hot(ground_truth_ind).to(torch.float)
                    ground_truth_zero = torch.zeros_like(ground_truth_one_hot)
                    ground_truth_uniform = torch.ones_like(ground_truth_one_hot) / num_sample_per_group
                    num_ori = 1 if self.args.random_select_cols else len(self.args.ori_text_cols)
                    num_neg = len(self.args.annotation_path) if self.args.random_select_cols else len(self.args.negative_text_cols) * len(self.args.annotation_path)
                    num_pos = len(self.args.annotation_path) if self.args.random_select_cols else len(self.args.positive_text_cols) * len(self.args.annotation_path)
                    if len(self.args.positive_text_cols) == 0:
                        num_pos = 0
                    total_num_pos = num_ori + num_pos
                    if self.image_group_number == 1: # only real images
                        if num_pos > 0:
                            if not self.args.sigmoid: # softmax
                                image_ground_truth = torch.cat([ground_truth_one_hot/total_num_pos, torch.cat([ground_truth_zero]*num_neg, dim=-1), torch.cat([ground_truth_one_hot/total_num_pos]*num_pos, dim=-1)], dim=-1)
                                text_ground_truth = torch.cat([ground_truth_one_hot, torch.cat([ground_truth_uniform]*num_neg, dim=0), torch.cat([ground_truth_one_hot]*num_pos, dim=0)], dim=0)
                            else: # sigmoid
                                image_ground_truth = torch.cat([torch.cat([ground_truth_one_hot]*num_ori, dim=-1), torch.cat([ground_truth_zero]*num_neg, dim=-1), torch.cat([ground_truth_one_hot]*num_pos, dim=-1)], dim=-1)
                                text_ground_truth = image_ground_truth.t()
                        else: # only negative samples
                            if not self.args.sigmoid:
                                image_ground_truth = torch.cat([ground_truth_one_hot/total_num_pos, torch.cat([ground_truth_zero]*num_neg, dim=-1)], dim=-1)
                                text_ground_truth = torch.cat([ground_truth_one_hot, torch.cat([ground_truth_uniform]*num_neg, dim=0)], dim=0)
                            else:
                                image_ground_truth = torch.cat([torch.cat([ground_truth_one_hot]*num_ori, dim=-1), torch.cat([ground_truth_zero]*num_neg, dim=-1)], dim=-1)
                                text_ground_truth = image_ground_truth.t()
                    else: # real + negative + positive
                        # for real
                        if num_pos > 0:
                            if not self.args.sigmoid:
                                image_ground_truth_0 = torch.cat([ground_truth_one_hot/total_num_pos, torch.cat([ground_truth_zero]*num_neg, dim=-1), torch.cat([ground_truth_one_hot/total_num_pos]*num_pos, dim=-1)], dim=-1)
                            else:
                                image_ground_truth_0 = torch.cat([torch.cat([ground_truth_one_hot]*num_ori, dim=-1), torch.cat([ground_truth_zero]*num_neg, dim=-1), torch.cat([ground_truth_one_hot]*num_pos, dim=-1)], dim=-1)
                        else:
                            if not self.args.sigmoid:
                                image_ground_truth_0 = torch.cat([ground_truth_one_hot/total_num_pos, torch.cat([ground_truth_zero]*num_neg, dim=-1)], dim=-1)
                            else:
                                image_ground_truth_0 = torch.cat([torch.cat([ground_truth_one_hot]*num_ori, dim=-1), torch.cat([ground_truth_zero]*num_neg, dim=-1)], dim=-1)
                        # for negative 
                        image_ground_truth_1 = torch.cat([torch.cat([ground_truth_zero]*self.text_group_number, dim=-1)]*num_neg, dim=0)
                        for ii in range(num_neg):
                            image_ground_truth_1[ii*num_sample_per_group:(ii+1)*num_sample_per_group,(ii+1)*num_sample_per_group:(ii+2)*num_sample_per_group] = ground_truth_one_hot
                        # for positive
                        image_ground_truth_2 = image_ground_truth_0.clone()
                        # combination
                        if num_pos > 0:
                            image_ground_truth = torch.cat([torch.cat([image_ground_truth_0]*num_ori, dim=0), image_ground_truth_1, torch.cat([image_ground_truth_2]*num_pos, dim=0)], dim=0)
                        else:
                            image_ground_truth = torch.cat([torch.cat([image_ground_truth_0]*num_ori, dim=0), image_ground_truth_1], dim=0)
                        text_ground_truth = image_ground_truth.t()
                ###
                ### margin loss
                if self.args.margin_weight > 0.0:
                    margin_loss = self.args.margin_weight * ada_margin_loss(logits_per_image, logits_per_text, self.args.margin_thre, self.image_group_number, self.text_group_number, additional_real_weight=self.args.margin_additional_real_weight, margin_low=self.args.margin_low, margin_scale=self.args.margin_scale)
                ###
                ### loss
                cls_loss = (loss_img(logits_per_image, image_ground_truth) + loss_txt(logits_per_text, text_ground_truth)) / 2
                ###
                ### total loss
                total_loss = cls_loss
                if self.args.margin_weight > 0.0:
                    total_loss += margin_loss
                ###
                ### log loss
                if self.args.local_rank == 0:
                    log_total_loss += total_loss.item()
                    log_loss_cls += cls_loss.item()
                ###
                total_loss.backward()
                if self.args.lora_r <= 0:
                    self.convert_models_to_fp32(self.model)
                model_optimizer.step()
                torch.distributed.barrier()
                if self.args.lora_r <= 0:
                    convert_weights(self.model)
                if self.args.local_rank == 0:
                    pbar.set_description(
                        f"epoch {epoch} lr {model_optimizer.param_groups[0]['lr']:.8f} batch: {i} - cls loss: {log_loss_cls / (i + 1):.5f}"
                    )
                torch.distributed.barrier()
                scheduler.step()
                global_step += 1
                if global_step >= self.args.stop_steps:
                    break

            if self.args.local_rank == 0:
                print(
                    "epoch: ",
                    epoch,
                    " - avg. loss: ",
                    log_total_loss / (i + 1),
                    " - lr: ",
                    model_optimizer.param_groups[0]["lr"],
                )
                all_losses.append(log_total_loss / (i + 1))
                all_lrs.append(scheduler.get_last_lr())

                save_path = self.construct_save_path(
                    starting_time, extension="ckpt"
                )
                torch.save(
                    {
                        "model": self.model.module.state_dict(),
                        "epoch": epoch,
                        "global_step": global_step,
                        "base_lr": self.args.base_lr,
                        "betas": (self.args.beta1, self.args.beta2),
                        "eps": self.args.eps,
                        "weight_decay": self.args.weight_decay,
                    },
                    save_path,
                )

                for plt_vals, suffix in zip([all_losses, all_lrs], ["loss", "LRs"]):
                    save_path = save_path = self.construct_save_path(
                        starting_time, suffix=suffix, extension="png"
                    )
                    plt.plot(range(len(plt_vals)), plt_vals)
                    plt.savefig(save_path)
                    plt.clf()
                    plt.cla()
                    plt.close()

            if global_step >= self.args.stop_steps:
                break

            torch.distributed.barrier()
        
        destroy_process_group()
