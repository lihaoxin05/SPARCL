import argparse
import src.core as core
import os
import torch
import torch.multiprocessing as mp

def parse_arguments():
    parser = argparse.ArgumentParser(description="FT CLIP MODEL WITH SYNTHETIC")
    ### model
    parser.add_argument(
        "--backbone_name",
        default="RN50",
        type=str,
        help="either of RN50, RN50x64, ViT-B/32",
    )
    parser.add_argument(
        "--lora_r", default=-1, type=int, help="use any number above 0 to activate LoRA"
    )
    parser.add_argument(
        "--load_from_path", default="", type=str, help="filename to load from"
    )
    parser.add_argument(
        "--sigmoid", action="store_true", 
    )
    parser.add_argument(
        "--sigmoid_logit_init", 
        default=[100.0, -30.0], # 1/\tau, b in paper
        type=float,
        nargs='+',
    )
    parser.add_argument(
        "--trainable_logit", action="store_true", 
    )
    ###
    ### save
    parser.add_argument(
        "--folder_name",
        default="",
        type=str,
    )
    parser.add_argument(
        "--tag",
        default="base",
        type=str,
    )
    ###
    ### training
    parser.add_argument(
        "--seed",
        dest="seed",
        default=0,
        type=int,
        help="define seed for random distribution of dataset",
    )
    parser.add_argument(
        "--base_lr",
        "--base_learning_rate",
        default=5e-7,
        type=float,
        metavar="LR",
        help="max learning rate",
    )
    parser.add_argument(
        "--beta1",
        default=0.9,
        type=float,
        metavar="B1",
        help="betas first hyperparameter",
    )
    parser.add_argument(
        "--beta2",
        default=0.999,
        type=float,
        metavar="B2",
        help="betas second hyperparameter",
    )
    parser.add_argument(
        "--eps", default=1e-6, type=float, metavar="EPS", help="epsilon hyperparameter"
    )
    parser.add_argument(
        "--weight_decay",
        default=0.001,
        type=float,
        metavar="WD",
        help="weight decay hyperparameter",
    )
    parser.add_argument(
        "-b",
        "--batch_size",
        default=200,
        type=int,
        metavar="N",
        help="mini-batch size (default: 100)",
    )
    parser.add_argument(
        "--epochs", default=201, type=int, help=""
    )
    parser.add_argument(
        "--stop_steps", default=10000, type=int, help=""
    )
    ###
    ### DDP
    parser.add_argument(
        "--workers", default=4, type=int, help="number of cpus for data loading"
    )
    parser.add_argument(
        "--local_rank",
        default=os.environ.get("LOCAL_RANK", 0),
        type=int,
        help="Please ignore and do not set this argument.",
    )
    parser.add_argument(
        "--master_port",
        default='10001',
        type=str,
    )
    ###
    ### data
    parser.add_argument(
        "--data_type",
        default="COCO_neg_pos",
        type=str,
        nargs='+',
        help="types of generated data: position, color, size, material, action, or any combination thereof",
    )
    parser.add_argument(
        "--data_path",
        default="",
        type=str,
        nargs='+',
    )
    parser.add_argument(
        "--heavy_aug",
        default=False,
        action="store_true",
        help="Use heavy data augmentation",
    )
    parser.add_argument(
        "--annotation_path",
        default="",
        type=str,
        nargs='+',
    )
    parser.add_argument(
        "--ori_text_cols",
        default=[],
        type=int,
        nargs='+',
    )
    parser.add_argument(
        "--negative_text_cols",
        default=[],
        type=int,
        nargs='+',
    )
    parser.add_argument(
        "--positive_text_cols",
        default=[],
        type=int,
        nargs='+',
    )
    parser.add_argument(
        "--random_select_cols", action="store_true", 
    )
    parser.add_argument(
        "--capsplit",
        default=False,
        action="store_true",
        help="Use caption splitting for training",
    )
    ## style transfer
    parser.add_argument(
        "--style_transfer",
        default=None,
        type=str,
        help="Style transfer mechanism. Currently supported are: 'adain'",
    )
    parser.add_argument(
        "--style_transfer_alpha", default=0.7, type=float,
    )
    parser.add_argument(
        "--mixstyle",
        default=False,
        action="store_true",
        help="Use mixstyle for training real life + synthetic data",
    )
    ## margin loss
    parser.add_argument(
        "--margin_weight", # \lambda in paper
        default=0.0,
        type=float,
    )
    parser.add_argument(
        "--margin_thre", # m_0 in paper
        default=0.0,
        type=float,
    )
    parser.add_argument(
        "--margin_additional_real_weight", # \alpha in paper
        default=0.0,
        type=float,
    )
    parser.add_argument(
        "--margin_low", # \beta in paper 
        default=-2.0,
        type=float,
    )
    parser.add_argument(
        "--margin_scale", # \gamma in paper
        default=1.0,
        type=float,
    )
    ##
    parser.add_argument(
        "--all_features", action="store_true", 
    )
    ###
    
    return parser.parse_args()


if __name__ == "__main__":
    # Parse arguments
    args = parse_arguments()
    args.world_size = torch.cuda.device_count()

    # Finetune
    mp.spawn(core.FT_CLIP, args=(args,), nprocs=args.world_size)
