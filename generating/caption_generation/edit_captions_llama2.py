import os
import argparse
import csv
import copy
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

model_name_or_path = "TheBloke/Llama-2-13B-chat-GPTQ"
# To use a different branch, change revision
# For example: revision="main"
model = AutoModelForCausalLM.from_pretrained(model_name_or_path, device_map="cuda", trust_remote_code=False, revision="main")
tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, use_fast=True)


def get_positive_output(input):
    prompt = f'''
    User: {input}
    Assistant: Sure! Here's my edit: 
    '''
    prompt_template=f'''[INST] <<SYS>>
    You are an assistant assigned to help a user edit a sentence that describes an image. Make a minor change to the sentence by randomly altering, omitting, inserting, or replacing one word or phrase. The new sentence must strictly retain the same meaning as the original sentence. Use the provided template and respond with a single, valid sentence.
    <</SYS>>
    {prompt}[/INST]
    '''

    input_ids = tokenizer(prompt_template, return_tensors='pt').input_ids.cuda()
    output = model.generate(inputs=input_ids, temperature=0.9, do_sample=True, top_p=0.9, top_k=100, max_new_tokens=128)
    output = tokenizer.decode(output[0]).split("[/INST]")[-1].split("</s>")[0]
    return output


def get_negative_output(input):
    prompt = f'''
    User: {input}
    Assistant: Sure! Here's my edit: 
    '''
    prompt_template=f'''[INST] <<SYS>>
    You are an assistant assigned to help a human user edit a given sentence that describes an image. Make a minor change to the sentence by randomly altering, omitting, inserting, or replacing one word or phrase. Although the change should be minor, it must result in a significant difference in the sentence's meaning, making it unable to describe the original image. Use the provided template and respond with a single, valid sentence.
    <</SYS>>
    {prompt}[/INST]
    '''

    input_ids = tokenizer(prompt_template, return_tensors='pt').input_ids.cuda()
    output = model.generate(inputs=input_ids, temperature=0.9, do_sample=True, top_p=0.9, top_k=100, max_new_tokens=128)
    output = tokenizer.decode(output[0]).split("[/INST]")[-1].split("</s>")[0]
    return output



def config():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_csv",
                        default="../captions_train2014.csv",
                        type=str,
                        help="Path to a csv file with captions and image ids/paths.")
    parser.add_argument("--save_path",
                        default="./captions_train2014_AddEditCation.csv",
                        type=str,
                        help="Path to a csv file with the additional negative captions and image ids/paths.")
    parser.add_argument("--start",
                        default=1,
                        type=int)
    parser.add_argument("--end",
                        default=-1,
                        type=int)
    parser.add_argument("--input_col_id",
                        default=-1,
                        type=int)
    parser.add_argument("--output_col_name",
                        default="edit_caption",
                        type=str)
    return parser.parse_args()


if __name__ == "__main__":
    args = config()
    f = open(args.dataset_csv)
    r = csv.reader(f, delimiter ='\t')
    row0 = next(r)
    row0.append(args.output_col_name)
    save = [row0]
    count = 0
    for item in r:
        count += 1
        if count >= args.start and (args.end == -1 or (args.end > 0 and count < args.end)):
            if count % 1000 == 0:
                print('Finish: ', count)
            try:
                caption = item[args.input_col_id]
                # positive or negative
                if 'pos' in args.output_col_name:
                    edit_caption = get_positive_output(caption)
                elif 'neg' in args.output_col_name:
                    edit_caption = get_negative_output(caption)
                else:
                    assert False
                # extract the generated caption
                if 'edit:' in edit_caption:
                    edit_caption = edit_caption.split('edit:')[-1].strip()
                else:
                    edit_caption = edit_caption.strip()
                if '\n' in edit_caption:
                    edit_caption = edit_caption.split('\n')[0].strip()
                else:
                    edit_caption = edit_caption.strip()
                if 'ser:' in edit_caption:
                    edit_caption = edit_caption.split('ser:')[-1].strip()
                else:
                    edit_caption = edit_caption.strip()
                print(caption)
                print('##############')
                print(edit_caption)
                print('#############################')
                print('#############################')
                new_item = copy.deepcopy(item)
                new_item.append(edit_caption)
                save.append(new_item)
            except:
                print(count)
                continue

    writer = csv.writer(open(args.save_path, 'w'), delimiter ='\t', lineterminator='\n')
    writer.writerows(save)

