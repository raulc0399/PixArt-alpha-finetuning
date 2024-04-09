import torch
from diffusers import PixArtAlphaPipeline, ConsistencyDecoderVAE, AutoencoderKL, Transformer2DModel
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
    pipe = PixArtAlphaPipeline.from_pretrained(MODEL_ID, torch_dtype=torch.float16, use_safetensors=True, device_map="auto")
    return pipe

def get_lora_pipeline():
    transformer = Transformer2DModel.from_pretrained(MODEL_ID, subfolder="transformer", torch_dtype=torch.float16, device_map="auto")
    transformer = PeftModel.from_pretrained(transformer, paths.get_peft_folder(), device_map="auto")
    pipe = PixArtAlphaPipeline.from_pretrained(MODEL_ID, transformer=transformer, torch_dtype=torch.float16, use_safetensors=True, device_map="auto")

    del transformer
    torch.cuda.empty_cache()

    return pipe

def generate_images(pipe, prefix):
    prompts = [
        "A small cactus with a happy face in the Sahara desert.",
        "cute dragon creature",
        "A family of four, with two parents and two children, having a picnic in a park. The sun is shining brightly, and there's a picnic basket filled with food. The family is sitting on a checkered blanket, laughing and enjoying their time together.",
        "A scientist working in a lab filled with futuristic gadgets and machines. The scientist is holding a test tube with a glowing liquid, and there's a robot assistant helping him in the background. The lab has large windows showing a cityscape outside.",
        "A group of friends at a bowling alley. They are taking turns bowling, with one person about to release the ball down the lane. There's excitement and competition in the air, and the scoreboard shows a close game.",
        "A classroom scene with a teacher and students. The teacher is standing in front of a blackboard, explaining a complex math problem, while the students are listening intently, some scratching their heads and others taking notes.",
        "A musician playing a guitar on stage at a rock concert. The crowd is cheering and waving their hands in the air, with lights flashing and a disco ball spinning overhead. The musician is passionately performing a solo, lost in the music.",
        "cartoon-like illustration of a crowd of people at a concert"
    ]

    output_dir = "./generated/"
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)

    current_time = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
    for i, prompt in enumerate(prompts):
        image = pipe(prompt, num_inference_steps=20).images[0]    

        file_name = os.path.join(output_dir, f"{current_time}_{prefix}_img_{i}")

        info_json = {
            "prompt": prompt,
            "model_id": MODEL_ID,
        }
        with open(f"{file_name}.json", "w") as json_file:
            json.dump(info_json, json_file)
        
        image.save(f"{file_name}.png")

if __name__ == "__main__":
    pipe = get_default_pipeline()
    generate_images(pipe, "default")

    del pipe
    torch.cuda.empty_cache()

    pipe = get_lora_pipeline()
    generate_images(pipe, "lora")