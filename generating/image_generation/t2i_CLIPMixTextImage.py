from diffusers import DiffusionPipeline
import torch
import PIL
import argparse
import csv
import json
import os
from transformers import CLIPImageProcessor, CLIPVisionModelWithProjection

feature_extractor = CLIPImageProcessor.from_pretrained("openai/clip-vit-large-patch14")

image_encoder = CLIPVisionModelWithProjection.from_pretrained(
    "openai/clip-vit-large-patch14",
    torch_dtype=torch.float16,
).to("cuda")

pipe = DiffusionPipeline.from_pretrained("SimianLuo/LCM_Dreamshaper_v7", torch_dtype=torch.float16).to("cuda")
pipe.safety_checker = None


def edit_image(prompt, source_prompt, init_image, mix_alg=-1, num_inference_steps=4, guidance_scale=7.5, strength=0.5):
    ### image features
    image = feature_extractor(init_image, return_tensors="pt").pixel_values.to("cuda")
    image_enc_hidden_states = image_encoder(image, output_hidden_states=True).image_embeds #(1,768)
    ### text features
    text_input_ids = pipe.tokenizer(
        prompt,
        padding="max_length",
        max_length=pipe.tokenizer.model_max_length,
        truncation=True,
        return_tensors="pt",
    ).input_ids
    prompt_embeds = pipe.text_encoder(text_input_ids.to("cuda"))[0]
    ### mix
    eos_ind = torch.nonzero(text_input_ids == 49407)
    if mix_alg != -1 and eos_ind.shape[0] > 1:
        if mix_alg == 0: # replace the style part of text embedding with image embedding
            replace_start_ind = eos_ind[1][1].item()
            mix_embeds = image_enc_hidden_states.unsqueeze(1)
            prompt_embeds[:,replace_start_ind:,:] = mix_embeds
            images = pipe(prompt_embeds=prompt_embeds, num_inference_steps=num_inference_steps, guidance_scale=guidance_scale, strength=strength).images
            # images = pipe(prompt_embeds=prompt_embeds, num_inference_steps=num_inference_steps, guidance_scale=guidance_scale, strength=strength, generator=torch.Generator().manual_seed(0)).images
        else:
            assert False, "Not Implemented."
    else:
        images = pipe(prompt_embeds=prompt_embeds, num_inference_steps=num_inference_steps, guidance_scale=guidance_scale, strength=strength).images
        # images = pipe(prompt_embeds=prompt_embeds, num_inference_steps=num_inference_steps, guidance_scale=guidance_scale, strength=strength, generator=torch.Generator().manual_seed(0)).images
    
    return images[0]

def config():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_csv",
                        default=None,
                        type=str)
    parser.add_argument("--dataset_json",
                        default=None,
                        type=str)
    parser.add_argument("--read_path",
                        default="./source",
                        type=str)
    parser.add_argument("--id_column",
                        default=1,
                        type=int)
    parser.add_argument("--ori_caption_column",
                        default=2,
                        type=int)
    parser.add_argument("--edit_caption_column",
                        default=3,
                        type=int)
    parser.add_argument("--format",
                        default="COCO",
                        type=str)
    parser.add_argument("--save_path",
                        default="./save",
                        type=str)
    parser.add_argument("--start",
                        default=1,
                        type=int)
    parser.add_argument("--end",
                        default=-1,
                        type=int)
    parser.add_argument("--num_inference_steps",
                        default=8,
                        type=int)
    parser.add_argument("--guidance_scale",
                        default=7.5,
                        type=float)
    parser.add_argument("--strength",
                        default=0.5,
                        type=float)
    parser.add_argument("--mix_alg",
                        default=-1,
                        type=int)
    return parser.parse_args()

if __name__ == "__main__":
    args = config()
    if args.dataset_csv is not None:
        f = open(args.dataset_csv)
        r = csv.reader(f, delimiter ='\t')
        row0 = next(r)
        count = 0
        for item in r:
            count += 1
            if count >= args.start and (args.end == -1 or (args.end > 0 and count < args.end)):
                if count % 1000 == 0:
                    print('Finish: ', count)
                image_id = int(item[0])
                id = int(item[args.id_column])
                source_prompt = item[args.ori_caption_column]
                edit_prompt = item[args.edit_caption_column]
                if args.format == 'COCO':
                    source_image = PIL.Image.open(os.path.join(args.read_path, 'COCO_train2014_{:012d}.jpg'.format(image_id))).convert('RGB')
                elif args.format == 'synthetic':
                    source_image = PIL.Image.open(os.path.join(args.read_path, '{:012d}_{:06d}.jpg'.format(image_id, id))).convert('RGB')
                else:
                    assert False
                edited_image = edit_image(prompt=edit_prompt, source_prompt=source_prompt, init_image=source_image, num_inference_steps=args.num_inference_steps, guidance_scale=args.guidance_scale, strength=args.strength, mix_alg=args.mix_alg) 
                edited_image.save(os.path.join(args.save_path, '{:012d}_{:06d}.jpg'.format(image_id, id)))
    elif args.dataset_json is not None:
        with open(args.dataset_json, 'r') as file:
            data = json.load(file)
        count = 0
        for item in data.keys():
            count += 1
            if count >= args.start and (args.end == -1 or (args.end > 0 and count < args.end)):
                if count % 100 == 0:
                    print('Finish: ', count)
                image_name = data[item]['filename']
                prompt = data[item]['caption']
                negative_prompt = data[item]['negative_caption']
                if args.format == 'COCO':
                    source_image = PIL.Image.open(os.path.join(args.read_path, 'COCO_val2014_{}'.format(image_name))).convert('RGB')
                elif args.format == 'synthetic':
                    source_image = PIL.Image.open(os.path.join(args.read_path, '{:012d}_{:06d}.jpg'.format(image_id, id))).convert('RGB')
                else:
                    assert False
                edited_image = edit_image(prompt=prompt, source_prompt=negative_prompt, init_image=source_image, num_inference_steps=args.num_inference_steps, guidance_scale=args.guidance_scale, strength=args.strength, mix_alg=args.mix_alg) 
                edited_image.save(os.path.join(args.save_path, '{}'.format(image_name)))
    else:
        assert False

