# SPARCL
This repo is the implementation of SPARCL (CVPR2025): [**Enhancing Vision-Language Compositional Understanding with Multimodal Synthetic Data**](https://arxiv.org/abs/2503.01167).

The codes are organized into two folders:
1. The [generating](generating) folder contains the code for generating synthetic samples.
2. The [finetuning](finetuning) folder contains the code for finetuning CLIP models.

## Datasets
### Training
We use the COCO-2014 dataset as the training data source. The dataset could be downloaded [here](https://cocodataset.org/#home).

### Evaluation
We conduct evaluation on four vision-language compositional understanding benchmarks.
1. [ARO](https://github.com/mertyg/vision-language-models-are-bows/tree/main)
2. [VL-CheckList](https://github.com/om-ai-lab/VL-CheckList/blob/main/DATASETS.md)
3. [SugarCrepe](https://github.com/RAIVNLab/sugar-crepe)
4. [SugarCrepe++](https://github.com/Sri-Harsha/scpp)

## Generating synthetic samples
For generating synthetic samples, please go to [generating](generating).

## Finetuning
For finetuning CLIP models, please go to [finetuning](finetuning).

## Acknowledgments
This repository is based on [SyViC](https://github.com/uvavision/SyViC). We sincerely thank the authors for their code.

## Citation
If you find our repository useful, please consider citing our paper.
```
@inproceedings{SPARCL-2025,
  author = {Haoxin Li and Boyang Li},
  title = {Enhancing Vision-Language Compositional Understanding with Multimodal Synthetic Data.},
  booktitle = {The IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
  year = {2025}
}
```

## Contact
If you have any problem please email me at haoxin003@e.ntu.edu.sg or lihaoxin05@gmail.com.
