export MODEL_NAME="stabilityai/stable-diffusion-xl-base-1.0"
export VAE_NAME="madebyollin/sdxl-vae-fp16-fix"
# export DATASET_NAME="lambdalabs/naruto-blip-captions"
export DATASET_NAME="inpaint-context/coco_test_function"
export CUDA_VISIBLE_DEVICES="0" 

# accelerate launch --config_file="./ds_3.yaml" --gpu_ids 0 train_text_to_image_sdxl.py \
  # --unet_onnx \
  # --unet_onnx_device_ids 0 \
accelerate launch --config_file="fsdp_2_model.yaml" --gpu_ids 0,1 train_text_to_image_sdxl.py \
  --pretrained_model_name_or_path=$MODEL_NAME \
  --pretrained_vae_model_name_or_path=$VAE_NAME \
  --enable_xformers_memory_efficient_attention \
  --gradient_checkpointing \
  --dataset_name=$DATASET_NAME \
  --resolution=512 \
  --center_crop \
  --random_flip \
  --proportion_empty_prompts=0.2 \
  --train_batch_size=1 \
  --gradient_accumulation_steps=4 \
  --max_train_steps=10000 \
  --use_8bit_adam \
  --learning_rate=1e-06 \
  --lr_scheduler="constant" \
  --lr_warmup_steps=0 \
  --mixed_precision="fp16" \
  --report_to="wandb" \
  --push_to_hub
  # --validation_prompt="a cute Sundar Pichai creature" \
  # --validation_epochs 5 \
  # --checkpointing_steps=5000 \
  # --output_dir="sdxl-naruto-model" \