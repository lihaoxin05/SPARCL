import os
import json
import subprocess
import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm
from torch.utils.data import Dataset
from easydict import EasyDict as edict
from torchvision.datasets.utils import download_url


class VG_Relation_Val(Dataset):
    def __init__(self, image_preprocess, root_dir='./ARO', download=False):
        '''
        image_preprocess: a function that takes in a PIL image and returns a tensor.
        root_dir: Directory for the VG-R dataset.
        download: Whether to download the dataset if it does not exist.
        '''
        self.root_dir = root_dir
        annotation_file = os.path.join(root_dir, "visual_genome_relation.json")
        image_dir = os.path.join(root_dir, "images")
        if not os.path.exists(image_dir):
            print("Image Directory for VG_Relation could not be found!")
            if download:
                self.download()
            else:
                raise RuntimeError("Please either download the dataset by letting `--download` or specify the correct directory.")
        
        if not os.path.exists(annotation_file):
            subprocess.call(["gdown", "--id", "1kX2iCHEv0CADL8dSO1nMdW-V0NqIAiP3", "--output", annotation_file])
        
        with open(annotation_file, "r") as f:
            self.dataset = json.load(f)

        np.random.seed(0)
        self.dataset = np.random.choice(self.dataset, size=int(len(self.dataset)*0.1), replace=False)
        
        self.all_relations = list()
        for item in self.dataset:
            item["image_path"] = os.path.join(image_dir, item["image_path"])
            self.all_relations.append(item["relation_name"])

        self.image_preprocess = image_preprocess

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        test_case = self.dataset[index]
        image = Image.open(test_case["image_path"]).convert('RGB')
        # Get the bounding box that contains the relation. This is to remove the irrelevant details in the scene.
        image = image.crop((test_case["bbox_x"], test_case["bbox_y"], test_case["bbox_x"] + test_case["bbox_w"], test_case["bbox_y"] + test_case["bbox_h"]))

        if self.image_preprocess is not None:
            image = self.image_preprocess(image)

        # Each test case has a correct and incorrect caption.
        true_caption = test_case["true_caption"]
        false_caption = test_case["false_caption"]
        item = edict({"image_options": [image], "caption_options": [false_caption, true_caption]})
        return item
    
    def download(self):
        os.makedirs(self.root_dir, exist_ok=True)
        image_zip_file = os.path.join(self.root_dir, "vgr_vga_images.zip")
        subprocess.call(["gdown", "--no-cookies", "1qaPlrwhGNMrR3a11iopZUT_GPP_LrgP9", "--output", image_zip_file])
        subprocess.call(["unzip", "vgr_vga_images.zip"], cwd=self.root_dir)

        
    def evaluate_scores(self, scores):
        """
        Scores: N x 1 x 2, i.e. first caption is the perturbed one, second is the positive one
        """
        if isinstance(scores, tuple):
            scores_i2t = scores[1]
            scores_t2i = scores[0] 
        else:
            scores_t2i = scores
            scores_i2t = scores

        preds = np.argmax(np.squeeze(scores_i2t, axis=1), axis=-1)
        correct_mask = (preds == 1)

        all_relations = np.array(self.all_relations)

        result_records = []
        # Log the accuracy of all relations
        for relation in np.unique(all_relations):
            relation_mask = (all_relations == relation)
            if relation_mask.sum() == 0:
                continue
            result_records.append({
                "Relation": relation,
                "Accuracy": correct_mask[relation_mask].mean(),
                "Count": relation_mask.sum(),
                "Dataset": "Visual Genome Relation"
            })
        
        symmetric = ['adjusting', 'attached to', 'between', 'bigger than', 'biting', 'boarding', 'brushing', 'chewing', 'cleaning', 'climbing', 'close to', 'coming from', 'coming out of', 'contain', 'crossing', 'dragging', 'draped over', 'drinking', 'drinking from', 'driving', 'driving down', 'driving on', 'eating from', 'eating in', 'enclosing', 'exiting', 'facing', 'filled with', 'floating in', 'floating on', 'flying', 'flying above', 'flying in', 'flying over', 'flying through', 'full of', 'going down', 'going into', 'going through', 'grazing in', 'growing in', 'growing on', 'guiding', 'hanging from', 'hanging in', 'hanging off', 'hanging over', 'higher than', 'holding onto', 'hugging', 'in between', 'jumping off', 'jumping on', 'jumping over', 'kept in', 'larger than', 'leading', 'leaning over', 'leaving', 'licking', 'longer than', 'looking in', 'looking into', 'looking out', 'looking over', 'looking through', 'lying next to', 'lying on top of', 'making', 'mixed with', 'mounted on', 'moving', 'on the back of', 'on the edge of', 'on the front of', 'on the other side of', 'opening', 'painted on', 'parked at', 'parked beside', 'parked by', 'parked in', 'parked in front of', 'parked near', 'parked next to', 'perched on', 'petting', 'piled on', 'playing', 'playing in', 'playing on', 'playing with', 'pouring', 'reaching for', 'reading', 'reflected on', 'riding on', 'running in', 'running on', 'running through', 'seen through', 'sitting behind', 'sitting beside', 'sitting by', 'sitting in front of', 'sitting near', 'sitting next to', 'sitting under', 'skiing down', 'skiing on', 'sleeping in', 'sleeping on', 'smiling at', 'sniffing', 'splashing', 'sprinkled on', 'stacked on', 'standing against', 'standing around', 'standing behind', 'standing beside', 'standing in front of', 'standing near', 'standing next to', 'staring at', 'stuck in', 'surrounding', 'swimming in', 'swinging', 'talking to', 'topped with', 'touching', 'traveling down', 'traveling on', 'tying', 'typing on', 'underneath', 'wading in', 'waiting for', 'walking across', 'walking by', 'walking down', 'walking next to', 'walking through', 'working in', 'working on', 'worn on', 'wrapped around', 'wrapped in', 'by', 'of', 'near', 'next to', 'with', 'beside', 'on the side of', 'around']
        df = pd.DataFrame(result_records)
        df = df[~df.Relation.isin(symmetric)]
        return df.Accuracy.mean()


class VG_Attribution_Val(Dataset):
    def __init__(self, image_preprocess, root_dir='./ARO', download=False):
        '''
        image_preprocess: a function that takes in a PIL image and returns a tensor.
        root_dir: Directory for the VG-A dataset.
        '''
        self.root_dir = root_dir
        annotation_file = os.path.join(root_dir, "visual_genome_attribution.json")
        image_dir = os.path.join(root_dir, "images")
        if not os.path.exists(image_dir):
            print("Image Directory for VG_Attribution could not be found!")
            if download:
                self.download()
            else:
                raise RuntimeError("Please either download the dataset by letting `--download` or specify the correct directory.")
        
        
        if not os.path.exists(annotation_file):
            subprocess.call(["gdown", "--id", "13tWvOrNOLHxl3Rm9cR3geAdHx2qR3-Tw", "--output", annotation_file])

        with open(annotation_file, "r") as f:
            self.dataset = json.load(f)

        np.random.seed(0)
        self.dataset = np.random.choice(self.dataset, size=int(len(self.dataset)*0.1), replace=False)
        
        for item in self.dataset:
            item["image_path"] = os.path.join(image_dir, item["image_path"])
        
        # Set of attributes in each test case
        self.all_attributes = [f"{item['attributes'][0]}_{item['attributes'][1]}" for item in self.dataset]
        self.image_preprocess = image_preprocess

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        test_case = self.dataset[index]
        image = Image.open(test_case["image_path"]).convert('RGB')
        # Get the bounding box that contains the relation. This is to remove the irrelevant details in the scene.
        image = image.crop((test_case["bbox_x"], test_case["bbox_y"], test_case["bbox_x"] + test_case["bbox_w"], test_case["bbox_y"] + test_case["bbox_h"]))

        if self.image_preprocess is not None:
            image = self.image_preprocess(image)

        # Each test case has a correct and incorrect caption.
        true_caption = test_case["true_caption"]
        false_caption = test_case["false_caption"]
        item = edict({"image_options": [image], "caption_options": [false_caption, true_caption]})
        return item
    
    def download(self):
        os.makedirs(self.root_dir, exist_ok=True)
        image_zip_file = os.path.join(self.root_dir, "vgr_vga_images.zip")
        subprocess.call(["gdown", "--no-cookies",  "1qaPlrwhGNMrR3a11iopZUT_GPP_LrgP9", "--output", image_zip_file])
        subprocess.call(["unzip", "vgr_vga_images.zip"], cwd=self.root_dir)

    
    def evaluate_scores(self, scores):
        """
        Scores: N x 1 x 2, i.e. first caption is the perturbed one, second is the positive one
        """
        if isinstance(scores, tuple):
            scores_i2t = scores[1]
            scores_t2i = scores[0] 
        else:
            scores_t2i = scores
            scores_i2t = scores

        preds = np.argmax(np.squeeze(scores_i2t, axis=1), axis=-1)
        correct_mask = (preds == 1)
        result_records = []
        all_attributes = np.array(self.all_attributes)
        for attr in np.unique(all_attributes):
            attr_mask = (all_attributes == attr)
            if attr_mask.sum() < 25:
                continue
            result_records.append({
                "Attributes": attr,
                "Accuracy": correct_mask[attr_mask].mean(),
                "Count": attr_mask.sum(),
                "Dataset": "Visual Genome Attribution"
            })
        df = pd.DataFrame(result_records)
        return df.Accuracy.mean()

