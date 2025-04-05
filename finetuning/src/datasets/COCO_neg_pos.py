import json
import os
import random
import csv

from PIL import Image, ImageOps
from torch.utils.data import Dataset


class COCO_neg_pos(Dataset):
    def __init__(
        self,
        annotation_path=[""],
        image_path=[""],
        ori_text_cols=[],
        negative_text_cols=[],
        positive_text_cols=[],
        random_select_cols=False,
        transform=None,
    ):
        self.num_annotations = len(annotation_path)

        # no synthetic image or use synthetic images; all real images with the same image_id are identical, so only 1
        assert len(image_path) == 1 or len(image_path) == (1 + (len(negative_text_cols) + len(positive_text_cols)) * self.num_annotations)
        if not random_select_cols:
            assert len(ori_text_cols) == 1
        else:
            assert len(ori_text_cols) > 1 and len(ori_text_cols) == len(negative_text_cols) and (len(ori_text_cols) == len(positive_text_cols) or len(positive_text_cols) == 0) # allow no positive

        self.random_select_cols = random_select_cols
        self.transform = transform

        self.ori_text_cols = ori_text_cols
        self.negative_text_cols = negative_text_cols
        self.positive_text_cols = positive_text_cols

        self.image_id_list = []
        self.image_list = [[] for ii in range(len(image_path))]
        self.text_list = [[] for ii in range(len(ori_text_cols) + len(negative_text_cols) * self.num_annotations + len(positive_text_cols) * self.num_annotations)] # only one original

        files = [open(path, 'r') for path in annotation_path]   
        try:
            readers = [csv.reader(file, delimiter ='\t') for file in files]
            for reader in readers:
                next(reader)
        
            for items in zip(*readers):
                image_id = int(items[0][0])
                for ii in range(1, self.num_annotations):
                    assert int(items[ii][0]) == image_id
                self.image_id_list.append('{:012d}'.format(image_id))
                for jj in range(len(image_path)):
                    if jj == 0:
                        # real image
                        self.image_list[jj].append(os.path.join(image_path[jj], 'COCO_train2014_{:012d}.jpg'.format(image_id)))
                    else:
                        # syn images
                        annotation_ind = ((jj - 1)//len(ori_text_cols))%self.num_annotations
                        ori_text_col_ind = (jj - 1)%len(ori_text_cols)
                        id = int(items[annotation_ind][ori_text_cols[ori_text_col_ind] - 1]) 
                        self.image_list[jj].append(os.path.join(image_path[jj], '{:012d}_{:06d}.jpg'.format(image_id, id)))
                # original text from only one file
                for jj in range(len(ori_text_cols)):
                    self.text_list[jj].append(items[0][ori_text_cols[jj]])
                for ii in range(self.num_annotations):
                    for jj in range(len(negative_text_cols)):
                        self.text_list[ii*len(negative_text_cols)+jj+len(ori_text_cols)].append(items[ii][negative_text_cols[jj]])
                for ii in range(self.num_annotations):
                    for jj in range(len(positive_text_cols)):
                        self.text_list[ii*len(positive_text_cols)+jj+len(ori_text_cols)+self.num_annotations*len(negative_text_cols)].append(items[ii][positive_text_cols[jj]])
        finally:
            for file in files:
                file.close()


    def __len__(self):
        return len(self.image_id_list)


    def __getitem__(self, idx):
        image = []
        text = []
        if not self.random_select_cols:
            # images
            for ii in range(len(self.image_list)):
                img_path = self.image_list[ii][idx]
                img = Image.open(img_path).convert("RGB")
                if self.transform:
                    img = self.transform(img)
                image.append(img)
            # texts
            for ii in range(len(self.text_list)):
                text.append(self.text_list[ii][idx])
        else:
            select_ind = random.choice(list(range(len(self.ori_text_cols))))
            # images
            if len(self.image_list) == 1:
                select_image_list_ind = [0]
            else:
                select_image_list_ind = [0]
                for anno_id in range(self.num_annotations):
                    select_image_list_ind.append(select_ind + 1 + anno_id * len(self.negative_text_cols))
                for anno_id in range(self.num_annotations):
                    select_image_list_ind.append(select_ind + 1 + self.num_annotations * len(self.negative_text_cols) + anno_id * len(self.positive_text_cols))
            for ind in select_image_list_ind:
                if ind >= len(self.image_list):
                    assert ind == select_image_list_ind[-1] # no positive ind
                else:
                    img_path = self.image_list[ind][idx]
                    img = Image.open(img_path).convert("RGB")
                    if self.transform:
                        img = self.transform(img)
                    image.append(img)
            # texts
            select_text_list_ind = [select_ind]
            for anno_id in range(self.num_annotations):
                select_text_list_ind.append(select_ind + len(self.ori_text_cols) + anno_id * len(self.negative_text_cols))
            for anno_id in range(self.num_annotations):
                select_text_list_ind.append(select_ind + len(self.ori_text_cols) + self.num_annotations * len(self.negative_text_cols) + anno_id * len(self.positive_text_cols))
            for ind in select_text_list_ind:
                if ind >= len(self.text_list):
                    assert ind == select_text_list_ind[-1] # no positive ind
                else:
                    text.append(self.text_list[ind][idx])

        return image, text
