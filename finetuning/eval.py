import torch
import numpy as np
from tqdm import tqdm
from collections import OrderedDict
import argparse

from src.lib.CLIP.clip import clip as clip

from src.datasets.CIFAR10 import CIFAR10
from src.datasets.ARO import VG_Relation, VG_Attribution
from src.datasets.ARO_val import VG_Relation_Val, VG_Attribution_Val
from src.datasets.SugarCrepe import SugarCrepe
from src.datasets.SugarCrepe_pp import SugarCrepe_pp
from src.datasets.VLCheckList import VLCheckList


def main(args):
    model, preprocess = clip.load(
        args.backbone_name,
        lora=args.lora_r,
        sigmoid=args.sigmoid,
    )
    if args.lora_r > 0:
        checkpoint = torch.load(args.checkpoint)
        state_dict = OrderedDict()
        for k, v in checkpoint['model'].items():
            if k[:7] == 'module.':
                name = k[7:]  # remove `module.`
            else:
                name = k
            state_dict[name] = v
        missing_keys, unexpected_keys = model.load_state_dict(state_dict)
        print('Missing keys: ', missing_keys)
        print('Unexpected keys: ', unexpected_keys)

    model.cuda()
    model.eval()

    if args.dataset == 'CIFAR10':
        dataset = CIFAR10(image_preprocess=preprocess, root_dir=args.dataset_path)
    elif args.dataset == 'VG_Relation':
        dataset = VG_Relation(image_preprocess=preprocess, download=True, root_dir=args.dataset_path)
    elif args.dataset == 'VG_Attribution':
        dataset = VG_Attribution(image_preprocess=preprocess, download=True, root_dir=args.dataset_path)
    elif args.dataset == 'VG_Relation_Val':
        dataset = VG_Relation_Val(image_preprocess=preprocess, download=True, root_dir=args.dataset_path)
    elif args.dataset == 'VG_Attribution_Val':
        dataset = VG_Attribution_Val(image_preprocess=preprocess, download=True, root_dir=args.dataset_path)
    elif args.dataset == 'SugarCrepe':
        dataset = SugarCrepe(image_preprocess=preprocess, root_dir=args.dataset_path, annotation_dir=args.annotation_path)
    elif args.dataset == 'SugarCrepe_pp':
        dataset = SugarCrepe_pp(image_preprocess=preprocess, root_dir=args.dataset_path, annotation_dir=args.annotation_path)
    elif args.dataset == 'VLCheckList':
        dataset = VLCheckList(image_preprocess=preprocess, root_dir=args.dataset_path, annotation_dir=args.annotation_path)
    else:
        assert False, "Not Implemented."
    data_loader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    scores = []
    tqdm_loader = tqdm(data_loader)
    tqdm_loader.set_description("Computing retrieval scores")

    if args.reuse_caption:
        reuse_count = 0

    with torch.inference_mode():
        for batch in tqdm_loader:
            image_options = []
            for i_option in batch["image_options"]: # number of image per sample
                image_embeddings = model.encode_image(i_option.cuda()).cpu().numpy() # B x D
                image_embeddings = image_embeddings / np.linalg.norm(image_embeddings, axis=1, keepdims=True) # B x D
                image_options.append(np.expand_dims(image_embeddings, axis=1))
            image_options = np.concatenate(image_options, axis=1) # B x K x D
            
            if not args.reuse_caption:
                caption_options = []
                for c_option in batch["caption_options"]: # number of caption per sample
                    caption_tokenized = torch.cat([clip.tokenize(c, truncate=True) for c in c_option])
                    caption_embeddings = model.encode_text(caption_tokenized.cuda()).cpu().numpy() # B x D
                    caption_embeddings = caption_embeddings / np.linalg.norm(caption_embeddings, axis=1, keepdims=True) # B x D
                    caption_options.append(np.expand_dims(caption_embeddings, axis=1))
                caption_options = np.concatenate(caption_options, axis=1) # B x L x D
            else:
                if reuse_count == 0:
                    if hasattr(dataset, 'candidate_templates'):
                        candidate_templates = dataset.candidate_templates
                    caption_options_ = []
                    for c_option in batch["caption_options"]: # number of caption per sample
                        if hasattr(dataset, 'candidate_templates'):
                            texts = [template.format(c_option[0]) for template in candidate_templates]
                        else:
                            texts = [c_option[0]]
                        caption_tokenized = clip.tokenize(texts, truncate=True) # (length of candidates) x 77
                        caption_embeddings = model.encode_text(caption_tokenized.cuda()).cpu().numpy() # (length of candidates) x D
                        caption_embeddings = caption_embeddings / np.linalg.norm(caption_embeddings, axis=1, keepdims=True) # (length of candidates) x D
                        if hasattr(dataset, 'candidate_templates'):
                            caption_embeddings = np.mean(caption_embeddings, axis=0, keepdims=True) # 1 x D
                            caption_embeddings = caption_embeddings / np.linalg.norm(caption_embeddings) # 1 x D
                        caption_options_.append(np.expand_dims(caption_embeddings, axis=1))
                    caption_options_ = np.concatenate(caption_options_, axis=1) # 1 x L x D
                    reuse_count += 1
                caption_options = np.concatenate([caption_options_]*image_options.shape[0], axis=0)
            batch_scores = np.einsum("nkd,nld->nkl", image_options, caption_options) # B x K x L
            scores.append(batch_scores)

    all_scores = np.concatenate(scores, axis=0) # N x K x L
    res = dataset.evaluate_scores(all_scores)
    print('Dataset: {}; Result: {}'.format(args.dataset, res))


def parse_args():
    parser = argparse.ArgumentParser(description="")
    parser.add_argument(
        "--backbone_name",
        default="RN50",
        type=str,
        help="either of RN50, RN50x64, ViT-B/32",
    )
    parser.add_argument(
        "--lora_r", 
        default=-1, 
        type=int, 
        help="use any number above 0 to activate LoRA"
    )
    parser.add_argument(
        "--sigmoid", action="store_true", 
    )
    parser.add_argument(
        "--checkpoint",
        default="",
        type=str,
    )
    parser.add_argument(
        "--dataset",
        default="",
        type=str,
    )
    parser.add_argument(
        "--dataset_path",
        default="",
        type=str,
    )
    parser.add_argument(
        "--annotation_path",
        default="",
        type=str,
    )
    parser.add_argument(
        "--batch_size", 
        default=256, 
        type=int,
    )
    parser.add_argument(
        "--num_workers", 
        default=8, 
        type=int,
    )
    parser.add_argument(
        "--reuse_caption", action="store_true", 
    )

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    main(args)