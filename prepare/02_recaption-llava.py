import torch
from transformers import BitsAndBytesConfig, AutoProcessor, LlavaForConditionalGeneration, LlavaNextProcessor, LlavaNextForConditionalGeneration
from PIL import Image

max_new_tokens = 200

this_dir = os.path.dirname(__file__)

train_folder = os.path.join(this_dir, "../../data/train/")
metadata_file_path = os.path.join(output_folder_path, "metadata.jsonl")

def get_quantization_config():
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
    )
    
    return quantization_config


def get_llava_model_and_processor(quantization_config):
    model_id = "llava-hf/llava-1.5-7b-hf"

    processor = AutoProcessor.from_pretrained(model_id)
    model = LlavaForConditionalGeneration.from_pretrained(model_id, quantization_config=quantization_config)

    # as per https://huggingface.co/docs/transformers/main/en/model_doc/llava#usage-tips
    prompt = "USER: <image>\nDescribe this image and its style in a very detailed manner\nASSISTANT:"
    
    return model, processor, prompt

def get_llava_next_model_and_processor(quantization_config):
    model_id = "llava-hf/llava-v1.6-mistral-7b-hf"

    processor = LlavaNextProcessor.from_pretrained(model_id)
    model = LlavaNextForConditionalGeneration.from_pretrained(model_id, quantization_config=quantization_config) 

    # as per https://huggingface.co/docs/transformers/main/en/model_doc/llava_next#usage-tips
    prompt = "[INST] <image>\nDescribe this image and its style in a very detailed manner [/INST]"
    
    return model, processor, prompt

def generate_text(model, processor, prompt, image):
    inputs = processor(prompt, image, return_tensors="pt").to("cuda:0")
    output = model.generate(**inputs, max_new_tokens=max_new_tokens)
    
    return processor.decode(output[0], skip_special_tokens=True)



images = [
    Image.open("output/0000.png"),
    Image.open("output/0001.png"),
    Image.open("output/0002.png")
]

quantization_config = get_quantization_config()

llava_model, llava_processor, prompt = get_llava_model_and_processor(quantization_config)

print("Llava Model")
for image in images:
    print(generate_text(llava_model, llava_processor, prompt, image))
    print("---------------------------------------------------")

del llava_model, llava_processor

llava_next_model, llava_next_processor, prompt = get_llava_next_model_and_processor(quantization_config)

print("Llava Next Model")

for image in images:
    print(generate_text(llava_next_model, llava_next_processor, prompt, image))
    print("---------------------------------------------------")
