# *[Specify the config file path and the GPU devices to use]
export CUDA_VISIBLE_DEVICES=0,1,2,3
export HF_ENDPOINT=https://hf-mirror.com
# *[Specify the config file path]
export OMINI_CONFIG=./train/config/sod.yaml

# *[Specify the WANDB API key]
# export WANDB_API_KEY='YOUR_WANDB_API_KEY'

echo $OMINI_CONFIG
export TOKENIZERS_PARALLELISM=true

accelerate launch \
  --num_processes 4 \
  --num_machines 1 \
  --mixed_precision bf16 \
  --dynamo_backend no \
  --main_process_port 41370 \
  -m omini.train_flux.train_sod2_multi_ddp