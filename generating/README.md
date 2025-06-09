# SPARCL
This is our implementation of generating synthetic samples. 


## Caption Generation
We use [Llama2](https://huggingface.co/docs/transformers/en/model_doc/llama2) to generate captions.
1. Create a conda environment using a config file [here](./caption_generation/llama2.yaml) and activate it.
2. Organize the COCO captions in one [file](https://1drv.ms/x/c/c88d845f827102a9/Eb4UnDljcVNGtBA9Gxc-mmUB79TThGreiNHOJnUssHGr3g?e=kgU2ZO).
3. Run the following commands to generate negative caption for one column.
```
cd caption_generation
CUDA_VISIBLE_DEVICES=0 python edit_captions_llama2.py \
--dataset_csv ./captions_train2014_unique_imageid_separate_ids_separate_captions.csv \
--save_path ./captions_train2014_unique_imageid_separate_ids_separate_captions_AddNegCaption_tmp.csv \
--start 1 --end 11 --input_col_id 2 --output_col_name neg_caption_1
```
4. Generate negative and positive captions for all columns.
5. Combine the generated captions in one file. We provide the file [here](https://1drv.ms/x/c/c88d845f827102a9/EV1kwU7U-mBNomGVdGzTOAoBZL2Z91Q8DUPEMbFiaPdokg?e=7cM1zG).


## Image Generation
We use [Latent Consistency Model](https://huggingface.co/docs/diffusers/en/using-diffusers/inference_with_lcm) to generate images.
1. Create a conda environment using a config file [here](./image_generation/diffuser-lcm.yaml) and activate it.
2. Run the following commands to generate images. Add `--mix_alg 0` to enable image feature injection in the generation.
```
cd image_generation
CUDA_VISIBLE_DEVICES=0 python t2i_CLIPMixTextImage.py \
--dataset_csv ./captions_train2014_unique_imageid_separate_ids_separate_captions_Add5NegCaption1_Add5PosCaption1.csv \
--id_column 1 --ori_caption_column 2 --edit_caption_column 3 \
--read_path /path/to/COCO/2014/train2014 \
--save_path ./ \
--start 1 --end 11 \
--mix_alg 0
```
3. We have provided some examples of generated images [here](https://1drv.ms/u/c/c88d845f827102a9/EcEEEG86W49PuflEC9T04wkB5OV2jKOUvz7XDoYvfp2hQA?e=BaAb2q).