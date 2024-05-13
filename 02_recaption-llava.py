import torch
from transformers import BitsAndBytesConfig, AutoProcessor, LlavaForConditionalGeneration, LlavaNextProcessor, LlavaNextForConditionalGeneration
from PIL import Image
import os
import pandas as pd
import paths

max_new_tokens = 200

train_folder = paths.get_train_folder()
metadata_file_path = paths.get_metadata_file_path()

# https://arxiv.org/pdf/2310.00426.pdf, Fig. 10
prompt_for_caption = "Describe this image and its style in a very detailed manner"
# prompt_for_caption_with_caption = """Give the caption of this image '{caption}', describe this image and its style in a very detailed manner,
# start all with 'Image in the style of simpsons cartoons, ', do not use the name simpsons in the description, do not use the name of the characters in the description.
# specify if the image contains one or more persons, animals, or objects, detail each of them, for persons and animals specify their emotions, actions, and interactions as well as if they are adults or children.
# """
prompt_for_caption_with_caption = """Give the caption of this image '{caption}', describe this image and its style in a very detailed manner,
start all with 'Image in lego city adventures style, ', do not use the name of the characters in the description.
specify if the image contains one or more persons, animals, or objects, detail each of them, for persons and animals specify their emotions, actions, and interactions as well as if they are adults or children.
"""

def get_quantization_config():
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16
    )
    
    return quantization_config

def get_llava_model_and_processor(quantization_config):
    model_id = "llava-hf/llava-1.5-7b-hf"

    processor = AutoProcessor.from_pretrained(model_id)
    model = LlavaForConditionalGeneration.from_pretrained(model_id, quantization_config=quantization_config)
    
    # as per https://huggingface.co/docs/transformers/main/en/model_doc/llava#usage-tips
    model_prompt = "USER: <image>\n{prompt_for_caption}\nASSISTANT:"
    model_eop = "ASSISTANT:"
    
    return model, processor, model_prompt, model_eop

def get_llava_next_model_and_processor(quantization_config):
    model_id = "llava-hf/llava-v1.6-mistral-7b-hf"
    
    processor = LlavaNextProcessor.from_pretrained(model_id)
    model = LlavaNextForConditionalGeneration.from_pretrained(model_id, quantization_config=quantization_config)
    
    # as per https://huggingface.co/docs/transformers/main/en/model_doc/llava_next#usage-tips
    model_prompt = "[INST] <image>\n{prompt_for_caption} [/INST]"
    model_eop = "[/INST]"
    
    return model, processor, model_prompt, model_eop

def generate_text(model, processor, prompt, image):
    inputs = processor(prompt, image, return_tensors="pt").to("cuda:0")
    output = model.generate(**inputs, max_new_tokens=max_new_tokens)
    
    return processor.decode(output[0], skip_special_tokens=True)

def save_generation_info(index, prompt1, prompt2, orig_caption, caption, caption_with_orig_caption, model_name, img_name):
    output_folder = "./check"
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    output_file_path = f"./{output_folder}/output_{model_name}_{index}.txt"

    with open(output_file_path, "w") as f:
        f.write(f"---------------------------: {index}\n")
        f.write(f"Image: {img_name}\n\n")
        f.write(f"Original Caption: {orig_caption}\n\n")

        f.write(f"Prompt: {prompt1}\n")
        f.write(f"\tCaption: {caption}\n\n")
        
        f.write(f"Prompt: {prompt2}\n")
        f.write(f"\tCaption with Original Caption: {caption_with_orig_caption}\n\n")


def generate_images_captions(df, model, processor, model_prompt, model_eop, model_name):
    length = len(df)
    for index, row in df.iterrows():
        print(f"Processing {index}/{length}")

        image_path = os.path.join(train_folder, row['file_name'])
        image = Image.open(image_path)

        # ignoring, use only the prompt with existing caption - generating caption using the given model and asking it to describe the image
        # prompt1 = model_prompt.format(prompt_for_caption=prompt_for_caption)                
        # caption = generate_text(model, processor, prompt1, image)
        prompt1 = ""
        caption = ""

        # df.at[index, f'{model_name}_caption'] = caption

        # generating caption using the given model and asking it to describe the image, also including the original caption
        orig_caption = row['orig_text']
        prompt_with_orig_caption = prompt_for_caption_with_caption.format(caption=orig_caption)
        prompt2 = model_prompt.format(prompt_for_caption=prompt_with_orig_caption)
        caption_with_orig_caption = generate_text(model, processor, prompt2, image)

        only_caption = caption_with_orig_caption.split(model_eop)[-1].strip()
        
        df.at[index, f'{model_name}_caption_with_orig_caption'] = only_caption

        if index < 10:
            save_generation_info(index, prompt1, prompt2, orig_caption, caption, caption_with_orig_caption, model_name, row['file_name'])

        if index % 20 == 0:
            df.to_json(metadata_file_path, orient='records', lines=True)

    df.to_json(metadata_file_path, orient='records', lines=True)
        
if __name__ == "__main__":
    metadata_df = pd.read_json(path_or_buf=metadata_file_path, lines=True)

    # remove the columns - used for regeneration of captions    
    metadata_df = metadata_df.drop(['llava_caption_with_orig_caption', 'llava_next_caption_with_orig_caption'], axis=1, errors='ignore')

    # sample 100
    # metadata_df = metadata_df.sample(n=100, ignore_index=True)
    
    quantization_config = get_quantization_config()

    print("generating captions with llava")
    llava_model, llava_processor, llava_model_prompt, llava_model_eop = get_llava_model_and_processor(quantization_config)
    generate_images_captions(metadata_df, llava_model, llava_processor, llava_model_prompt, llava_model_eop, 'llava')

    del llava_model, llava_processor, llava_model_prompt, llava_model_eop

    # print("\ngenerating captions with llava next")
    # llava_next_model, llava_next_processor, llava_next_prompt, llava_next_model_eop = get_llava_next_model_and_processor(quantization_config)
    # generate_images_captions(metadata_df, llava_next_model, llava_next_processor, llava_next_prompt, llava_next_model_eop, 'llava_next')
