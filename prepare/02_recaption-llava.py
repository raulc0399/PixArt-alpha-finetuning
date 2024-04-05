import torch
from transformers import BitsAndBytesConfig, AutoProcessor, LlavaForConditionalGeneration, LlavaNextProcessor, LlavaNextForConditionalGeneration
from PIL import Image

quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16
)

# llava
model_id = "llava-hf/llava-1.5-7b-hf"

processor = AutoProcessor.from_pretrained(model_id)
model = LlavaForConditionalGeneration.from_pretrained(model_id, quantization_config=quantization_config, device_map="auto")

max_new_tokens = 200
prompt = "USER: <image>\nDescribe this image and its style in a very detailed manner\nASSISTANT:"

image = Image.open("output/0000.png")
image = Image.open("output/0000.png")
image1 = Image.open("output/0001.png")
image2 = Image.open("output/0002.png")

prompt = "<image>\nUSER: Describe this image and its style in a very detailed manner\nASSISTANT:"
prompt1 = "<image>\nUSER: Describe this image and its style in a very detailed manner\nASSISTANT:"
prompt2 = "<image>\nUSER: Describe this image and its style in a very detailed manner\nASSISTANT:"

# inputs = processor(prompt, image, padding=True, return_tensors="pt").to("cuda")
inputs = processor([prompt, prompt1, prompt2], images=[image, image1, image2], padding=True, return_tensors="pt").to("cuda")

outputs = model.generate(**inputs, max_new_tokens=200)

# print(processor.decode(outputs[0], skip_special_tokens=True))
generated_text = processor.batch_decode(outputs, skip_special_tokens=True)
for text in generated_text:
    print(text)
    print("---------------------------------------------------")

exit()

# llava-next
processor = LlavaNextProcessor.from_pretrained("llava-hf/llava-v1.6-mistral-7b-hf")

model = LlavaNextForConditionalGeneration.from_pretrained("llava-hf/llava-v1.6-mistral-7b-hf", torch_dtype=torch.float16, quantization_config=quantization_config, low_cpu_mem_usage=True) 

image = Image.open("output/0000.png")
image1 = Image.open("output/0001.png")
image2 = Image.open("output/0002.png")

prompt = "[INST] <image>\nDescribe this image and its style in a very detailed manner? [/INST]"
prompt1 = "[INST] <image>\nDescribe this image and its style in a very detailed manner? [/INST]"
prompt2 = "[INST] <image>\nDescribe this image and its style in a very detailed manner? [/INST]"

# inputs = processor(prompt, image, return_tensors="pt").to("cuda:0")

# padding=True, otherwise: RuntimeError: stack expects each tensor to be equal size, but got [1326, 4096] at entry 0 and [1526, 4096] at entry 1
inputs = processor([prompt, prompt1, prompt2], images=[image, image1, image2], padding=True, return_tensors="pt").to("cuda")

output = model.generate(**inputs, max_new_tokens=200)

# print(processor.decode(output[0], skip_special_tokens=True))
generated_text = processor.batch_decode(output, skip_special_tokens=True)
for text in generated_text:
    print(text)
    print("---------------------------------------------------")

