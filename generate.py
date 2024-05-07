import torch
from diffusers import PixArtAlphaPipeline, Transformer2DModel
from transformers import T5EncoderModel
from peft import PeftModel
import datetime
import json
import os
import paths

MODEL_ID = "PixArt-alpha/PixArt-XL-2-512x512"
# MODEL_ID = "PixArt-alpha/PixArt-XL-2-1024-MS"

# If use DALL-E 3 Consistency Decoder
# pipe.vae = ConsistencyDecoderVAE.from_pretrained("openai/consistency-decoder", torch_dtype=torch.float16)

# If use SA-Solver sampler
# from diffusion.sa_solver_diffusers import SASolverScheduler
# pipe.scheduler = SASolverScheduler.from_config(pipe.scheduler.config, algorithm_type='data_prediction')

# If loading a LoRA model
# transformer = Transformer2DModel.from_pretrained("PixArt-alpha/PixArt-LCM-XL-2-1024-MS", subfolder="transformer", torch_dtype=torch.float16)
# transformer = PeftModel.from_pretrained(transformer, "Your-LoRA-Model-Path")
# pipe = PixArtAlphaPipeline.from_pretrained("PixArt-alpha/PixArt-LCM-XL-2-1024-MS", transformer=transformer, torch_dtype=torch.float16, use_safetensors=True)
# del transformer

def get_default_pipeline():
    pipe = PixArtAlphaPipeline.from_pretrained(MODEL_ID, torch_dtype=torch.float16)
    pipe.to("cuda")
    return pipe

def get_lora_pipeline():
    transformer = Transformer2DModel.from_pretrained(MODEL_ID, subfolder="transformer", torch_dtype=torch.float16)

    perf_ckpt = paths.get_transformer_peft_folder()
    # perf_ckpt = os.path.join(perf_ckpt, "checkpoint-7100")

    transformer = PeftModel.from_pretrained(transformer, perf_ckpt)

    text_encoder = T5EncoderModel.from_pretrained(MODEL_ID, subfolder='text_encoder', torch_dtype=torch.float16)
    text_encoder = PeftModel.from_pretrained(text_encoder, paths.get_text_encoder_peft_folder())
    
    pipe = PixArtAlphaPipeline.from_pretrained(MODEL_ID, 
                                               transformer=transformer,
                                               text_encoder=text_encoder,
                                               torch_dtype=torch.float16)
    pipe.to("cuda")

    # del transformer
    # del text_encoder
    # torch.cuda.empty_cache()

    return pipe

def generate_image(pipe, prompt, i, current_time, prefix, output_dir):
    image = pipe(prompt, num_inference_steps=20).images[0]

    file_name = os.path.join(output_dir, f"{current_time}_{prefix}_img_{i}")

    info_json = {
        "prompt": prompt,
        "model_id": MODEL_ID,
    }
    with open(f"{file_name}.json", "w") as json_file:
        json.dump(info_json, json_file)
    
    image.save(f"{file_name}.png")

def generate_images(pipe, prefix):
    prompts = [
        "beautiful lady, freckles, big smile, blue eyes, short ginger hair, dark makeup, wearing a floral blue vest top, soft light, dark grey background",
        "friends hanging out on a beautiful summer evening, beach bar.",
        "teacher explaining, in fun and entertaining way, physics to a group of interested kids.",
        "crowd at a concert, enjoying the music and the atmosphere.",
        "a small cactus with a happy face in the Sahara desert.",
        "professional portrait photo of an anthropomorphic cat wearing fancy gentleman hat and jacket walking in autumn forest.",
        "cute dragon creature",
        "a very beautiful and colorful bird",
        "a cute little puppy",
        "a big football stadium",
        "a house in a modern city on a sunny day",
        "Pirate ship trapped in a cosmic maelstrom nebula, rendered in cosmic beach whirlpool engine, volumetric lighting, spectacular, ambient lights, light pollution, cinematic atmosphere, art nouveau style, illustration art artwork by SenseiJaye, intricate detail."
    ]

    output_dir = "./generated/"
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)

    current_time = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
    for i, prompt in enumerate(prompts):
        generate_image(pipe, prompt, i, current_time, prefix, output_dir)

        if prefix == "lora":
            prompt = f"Image in the style of simpsons cartoons, {prompt}"

            generate_image(pipe, prompt, i, current_time, f"{prefix}_w_trigger", output_dir)

if __name__ == "__main__":
    if True:
        pipe = get_default_pipeline()
        generate_images(pipe, "default")

        del pipe
        torch.cuda.empty_cache()

    pipe = get_lora_pipeline()
    generate_images(pipe, "lora")