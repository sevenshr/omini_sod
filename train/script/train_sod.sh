# *[Specify the config file path and the GPU devices to use]
# export CUDA_VISIBLE_DEVICES=0,1
export HF_ENDPOINT=https://hf-mirror.com
# *[Specify the config file path]
export OMINI_CONFIG=./train/config/sod.yaml

# *[Specify the WANDB API key]
# export WANDB_API_KEY='YOUR_WANDB_API_KEY'

echo $OMINI_CONFIG
export TOKENIZERS_PARALLELISM=true

accelerate launch --main_process_port 41353 -m omini.train_flux.train_sod
