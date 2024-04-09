#!/bin/bash

# clone original repo and then run the training script
# git clone https://github.com/PixArt-alpha/PixArt-alpha.git

# also run
# accelerate config

# check with
# accelerate env

# remove the validation_epochs or set it to a lower number if you want to run the validation prompt
# validation will be ran at the end
accelerate launch --num_processes=1 --main_process_port=36667 PixArt-alpha/train_scripts/train_pixart_lora_hf.py --mixed_precision="fp16" \
  --pretrained_model_name_or_path=PixArt-alpha/PixArt-XL-2-512x512 \
  --train_data_dir="../data/train/" --caption_column="llava_caption_with_orig_caption" \
  --resolution=512 --random_flip \
  --train_batch_size=4 --gradient_accumulation_steps=4 \
  --num_train_epochs=4 --checkpointing_steps=25 \
  --learning_rate=1e-06 --lr_scheduler="constant" --lr_warmup_steps=0 \
  --seed=42 \
  --output_dir="pixart-simpson-model" \
  --report_to="tensorboard" \
  --gradient_checkpointing --checkpoints_total_limit=10 \
  --validation_epochs=1000 \
  --validation_prompt="cute dragon creature" \
  --rank=16
