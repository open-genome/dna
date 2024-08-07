#!/bin/bash

# Seeds array
seeds=(2222 42 43 44 45)

# Output file
output_file="experiment_output.txt"

# Loop over seeds
for seed in "${seeds[@]}"
do
    echo "Running experiment with seed $seed"
    echo "Running experiment with seed $seed" >> $output_file
    CUDA_VISIBLE_DEVICES='1' /mnt/nas/share2/home/by/envs/hndna/bin/python train.py experiment='hg38/genomic_benchmark_load_finetuned_model' wandb=null train.seed=$seed dataset.dest_path='/home/yubo/hyena-dna/data/genomic_benchmark' >> $output_file
done

echo "All experiments completed."

