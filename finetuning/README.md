# SPARCL
This is our implementation for fine-tuning CLIP models using multimodal synthetic data.


## Preparation
1. Create a conda environment using a config file [here](./syvic.yaml) and activate it.
2. Download [AdaIN models](https://github.com/naoto0804/pytorch-AdaIN/releases/tag/v0.0.0) and place them in the `./adain/models` directory.

## Training and Evaluation
To start training and evaluation, run the following commands:
```
cd scripts
bash run.sh
```
The `run.sh` script handles both training and evaluation. You can modify the script to set your desired hyper-parameters.

We release our trained model [here](https://1drv.ms/u/c/c88d845f827102a9/EedRYJSftYdDnr_872E313kBuNQSuR-hVpPNRmNCqk1zJg?e=t2lEPm).