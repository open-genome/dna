#!/bin/bash
export TQDM_SILENT=1
export TQDM_DISABLE=1
export CUDA_VISIBLE_DEVICES=1,2

# Seeds array
seeds=(2222 42 43 44 45 46 47 48 49 50)
# seeds=(46 47 48 49 50)
# Output file
output_file="experiment_output_ablation.txt"

# Loop over seeds
for seed in "${seeds[@]}"
do
    echo ">>>>>Running experiment with seed $seed<<<<<"
    echo "Running experiment with seed $seed" >> $output_file
    /mnt/nas/share2/home/by/envs/hndna/bin/python train.py experiment='hg38/nt_ablation' wandb=null train.seed=$seed >> $output_file
done

echo "All experiments completed."