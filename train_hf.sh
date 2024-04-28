#!/bin/bash

# clone original repo and then run the training script
# git clone https://github.com/PixArt-alpha/PixArt-alpha.git

# the patch loads the vae and tokenizer directly in fp16 on GPU - also the training of the transformers gives an error if using fp16, so that is removed
# to apply the patch:
# cp train_pixart_lora_hf.patch PixArt-alpha
# cd PixArt-alpha
# git apply train_pixart_lora_hf.patch

# also run
# accelerate config

# check with
# accelerate env

# remove the validation_epochs or set it to a lower number if you want to run the validation prompt
# validation will be ran at the end
accelerate launch --num_processes=1 --main_process_port=36667 PixArt-alpha/train_scripts/train_pixart_lora_hf.py --mixed_precision="fp16" \
  --pretrained_model_name_or_path=PixArt-alpha/PixArt-XL-2-512x512 \
  --train_data_dir="../data/train/" --caption_column="llava_caption_with_orig_caption" \
  --resolution=512 \
  --train_batch_size=2 --gradient_accumulation_steps=1 \
  --num_train_epochs=100 --checkpointing_steps=100 \
  --max_train_samples=400 \
  --learning_rate=3e-04 --lr_scheduler="cosine" --lr_warmup_steps=0 \
  --seed=42 \
  --output_dir="pixart-simpsons-model" \
  --report_to="wandb" \
  --gradient_checkpointing \
  --validation_epochs=5 \
  --validation_prompt="Image in the style of simpsons cartoons, cute dragon creature" \
  --rank=16 \
  --adam_weight_decay=0.03 --adam_epsilon=1e-10 \
  --dataloader_num_workers=8
    # --snr_gamma=1.0
  # --use_rslora
  # --use_dora

# https://github.com/huggingface/diffusers/tree/main/examples/text_to_image#training-with-min-snr-weighting

# https://ngwaifoong92.medium.com/how-to-fine-tune-stable-diffusion-using-lora-85690292c6a8
# https://huggingface.co/docs/diffusers/v0.27.2/en/training/text2image
# https://huggingface.co/docs/diffusers/v0.27.2/en/training/sdxl
# https://huggingface.co/docs/diffusers/training/lora
# https://github.com/bmaltais/kohya_ss/wiki/LoRA-training-parameters

# to try:
# epochs
# lora layers
# rank
# learning rate: 1e-06, 1e-05, 1e-04
# lr_scheduler: constant, linear, cosine
# lr_warmup_steps
# resolution - 512, 256, 300
# gradient clipping
# orig_text column
# scale_lr
# use_dora=True https://github.com/huggingface/peft/releases/tag/v0.9.0
# proportion_empty_prompts
# lora_dropout

# tries
# 1. 1e-06, constant
# 2. 1e-06, linear
# 3. 1e-05, constant https://keras.io/examples/generative/finetune_stable_diffusion/#initialize-the-trainer-and-compile-it
# 4. 1e-05, cosine
# 5. 1e-06, constant, num_train_epochs=75, lr_warmup_steps=50, train_batch_size=2, 8bit adam, orig_text, checkpointing_steps=500
