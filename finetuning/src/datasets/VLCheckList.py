import os
from torch.utils.data import Dataset
from PIL import Image
import json
from easydict import EasyDict as edict
import numpy as np
import yaml

class VLCheckList(Dataset):

    def __init__(self, root_dir, annotation_dir, image_preprocess=None):
        self.root = root_dir
        self.transform = image_preprocess
        self.ann = {}
        # self.dataset_ind = []
        key = 0
        all_yaml = os.listdir(annotation_dir)
        for i in range(len(all_yaml)):
            file_path = os.path.join(annotation_dir, all_yaml[i])
            anno_dict = yaml.load(open(file_path), Loader=yaml.FullLoader)
            anno_path = os.path.join(annotation_dir.split('corpus')[0], anno_dict['ANNO_PATH'])
            anno = json.load(open(anno_path))
            for ii in range(len(anno)):
                image_path = os.path.join(anno_dict['IMG_ROOT'], anno[ii][0])
                pos_caption = anno[ii][1]['POS']
                ### https://github.com/om-ai-lab/VL-CheckList/blob/main/vl_checklist/evaluate.py select 0-th
                # assert len(pos_caption) == 1, anno_path
                pos_caption = pos_caption[0]
                neg_caption = anno[ii][1]['NEG']
                # assert len(neg_caption) == 1, anno_path
                neg_caption = neg_caption[0]
                self.ann[str(key)] = {'filename': image_path, 'caption':pos_caption, 'negative_caption':neg_caption}
                # self.dataset_ind.append(i)
                key += 1
        self.keys = list(self.ann.keys())
        

    def __len__(self):
        return len(self.ann)

    def __getitem__(self, idx):
        data = self.ann[self.keys[idx]]
        img = Image.open(os.path.join(self.root, data['filename'])).convert('RGB')
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
        # dataset_ind = np.array(self.dataset_ind)
        # acc = []
        # for i in range(np.max(dataset_ind)+1):
        #     dataset_mask = (dataset_ind == i)
        #     acc.append(np.sum(correct_mask * dataset_mask) / np.sum(dataset_mask))
        return np.mean(correct_mask)