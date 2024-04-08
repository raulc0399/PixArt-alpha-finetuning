import torch
from diffusers import PixArtAlphaPipeline, ConsistencyDecoderVAE, AutoencoderKL
import datetime
import json
import os

# You can replace the checkpoint id with "PixArt-alpha/PixArt-XL-2-512x512" too.
MODEL_ID = "PixArt-alpha/PixArt-XL-2-512x512"
# MODEL_ID = "PixArt-alpha/PixArt-XL-2-1024-MS"

pipe = PixArtAlphaPipeline.from_pretrained(MODEL_ID, torch_dtype=torch.float16, use_safetensors=True, device_map="auto")

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

prompts = [
    "A small cactus with a happy face in the Sahara desert.",
    "cute dragon creature",
    "A family of four, with two parents and two children, having a picnic in a park, drawn in the Simpson style. The sun is shining brightly, and there's a picnic basket filled with food. The family is sitting on a checkered blanket, laughing and enjoying their time together.",
    "A scientist working in a lab filled with futuristic gadgets and machines, all rendered in the Simpson style. The scientist is holding a test tube with a glowing liquid, and there's a robot assistant helping him in the background. The lab has large windows showing a cityscape outside.",
    "A group of friends at a bowling alley, drawn in the Simpson style. They are taking turns bowling, with one person about to release the ball down the lane. There's excitement and competition in the air, and the scoreboard shows a close game.",
    "A classroom scene with a teacher and students, all rendered in the Simpson style. The teacher is standing in front of a blackboard, explaining a complex math problem, while the students are listening intently, some scratching their heads and others taking notes.",
    "A musician playing a guitar on stage at a rock concert, drawn in the Simpson style. The crowd is cheering and waving their hands in the air, with lights flashing and a disco ball spinning overhead. The musician is passionately performing a solo, lost in the music.",
    "A family of four, with two parents and two children, having a picnic in a park. The sun is shining brightly, and there's a picnic basket filled with food. The family is sitting on a checkered blanket, laughing and enjoying their time together.",
    "A scientist working in a lab filled with futuristic gadgets and machines. The scientist is holding a test tube with a glowing liquid, and there's a robot assistant helping him in the background. The lab has large windows showing a cityscape outside.",
    "A group of friends at a bowling alley. They are taking turns bowling, with one person about to release the ball down the lane. There's excitement and competition in the air, and the scoreboard shows a close game.",
    "A classroom scene with a teacher and students. The teacher is standing in front of a blackboard, explaining a complex math problem, while the students are listening intently, some scratching their heads and others taking notes.",
    "A musician playing a guitar on stage at a rock concert. The crowd is cheering and waving their hands in the air, with lights flashing and a disco ball spinning overhead. The musician is passionately performing a solo, lost in the music.",
]

output_dir = "./generated/"
if not os.path.exists(output_dir):
    os.mkdir(output_dir)

current_time = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
for i, prompt in enumerate(prompts):
    image = pipe(prompt).images[0]    

    file_name = os.path.join(output_dir, f"img_{i}_{current_time}")

    info_json = {
        "prompt": prompt,
        "model_id": MODEL_ID,
    }
    with open(f"{file_name}.json", "w") as json_file:
        json.dump(info_json, json_file)
    
    image.save(f"{file_name}.png")
