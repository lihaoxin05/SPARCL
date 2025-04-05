import os
import json
import numpy as np
from PIL import Image
import torchvision
from torch.utils.data import Dataset
from easydict import EasyDict as edict
import xml.etree.ElementTree as ET

### classes and templates are from https://github.com/openai/CLIP/blob/main/data/prompts.md#cifar10

classes = [
    'airplane',
    'automobile',
    'bird',
    'cat',
    'deer',
    'dog',
    'frog',
    'horse',
    'ship',
    'truck',
]

templates = [
    'a photo of a {}.',
    'a blurry photo of a {}.',
    'a black and white photo of a {}.',
    'a low contrast photo of a {}.',
    'a high contrast photo of a {}.',
    'a bad photo of a {}.',
    'a good photo of a {}.',
    'a photo of a small {}.',
    'a photo of a big {}.',
    'a photo of the {}.',
    'a blurry photo of the {}.',
    'a black and white photo of the {}.',
    'a low contrast photo of the {}.',
    'a high contrast photo of the {}.',
    'a bad photo of the {}.',
    'a good photo of the {}.',
    'a photo of the small {}.',
    'a photo of the big {}.',
]

class CIFAR10(Dataset):
    def __init__(self, image_preprocess, root_dir='./CIFAR10'):
        self.image_preprocess = image_preprocess
        self.candidate_templates = templates
        self.dataset = torchvision.datasets.CIFAR10(root=root_dir, download=True, train=False)
        print("All samples: ", len(self.dataset))

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        image, _ = self.dataset[index]

        if self.image_preprocess is not None:
            image = self.image_preprocess(image)

        item = edict({"image_options": [image], "caption_options": classes})
        return item
    
    def evaluate_scores(self, scores):
        if isinstance(scores, tuple):
            scores_i2t = scores[1]
            scores_t2i = scores[0] 
        else:
            scores_t2i = scores
            scores_i2t = scores

        preds = list(np.argmax(np.squeeze(scores_i2t, axis=1), axis=-1))
        gts = [self.dataset[i][1] for i in range(self.__len__())]
        assert len(preds) == len(gts)
        count = 0
        for i in range(self.__len__()):
            if preds[i] == gts[i]:
                count += 1
        return count / self.__len__()
