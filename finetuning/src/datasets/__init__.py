from src.datasets.COCO_neg_pos import COCO_neg_pos

dataset_types = {
    "COCO_neg_pos": COCO_neg_pos,
}

if __name__ == "__main__":
    from torch.utils.data import DataLoader, ConcatDataset
    from torchvision.transforms import ToTensor
    from torch.utils.data import ConcatDataset

    transform = ToTensor()
    dataset = TDW_Size(transform=transform)
    dataset = ConcatDataset(
        [ds(transform=transform) for ds in dataset_types.values()]
    )
    loader = DataLoader(dataset, drop_last=True)
    breakpoint()
