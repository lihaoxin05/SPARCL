import os
from torch.utils.data import Dataset
from PIL import Image
import json
from easydict import EasyDict as edict
import numpy as np

class SugarCrepe(Dataset):

    def __init__(self, root_dir, annotation_dir, image_preprocess=None):
        self.root = root_dir
        self.ann = json.load(open(annotation_dir))
        self.keys = list(self.ann.keys())
        self.transform = image_preprocess

    def __len__(self):
        return len(self.ann)

    def __getitem__(self, idx):
        data = self.ann[self.keys[idx]]
        img = Image.open(os.path.join(self.root, 'COCO_val2014_' + data['filename']))
        if self.transform is not None:
            img = self.transform(img)
        caption = data['caption']
        negative_caption = data['negative_caption']
        item = edict({"image_options": [img], "caption_options": [caption, negative_caption]})
        return item

    def evaluate_scores(self, scores):
        if isinstance(scores, tuple):
            scores_i2t = scores[0]
            scores_t2i = scores[1].T # Make it N_ims x N_text
        else:
            scores_t2i = scores
            scores_i2t = scores

        preds = np.argmax(np.squeeze(scores_i2t, axis=1), axis=-1)
        correct_mask = (preds == 0)
        return np.mean(correct_mask)