#!/bin/bash

# Seeds array
seeds=(2222 42 43 44 45 46 47 48 49 50)

# Output file
output_file="experiment_output_caduceus.txt"

# Loop over seeds
for seed in "${seeds[@]}"
do
    echo "Running experiment with seed $seed"
    echo "Running experiment with seed $seed" >> $output_file
    /mnt/nas/share2/home/by/envs/hndna/bin/python train.py experiment='hg38/genomic_benchmark_hyena' wandb=null train.seed=$seed dataset.dest_path='/mnt/nas/share2/home/by/hyena-dna/data/genomic_benchmark' dataset.dataset_name='human_ocr_ensembl' dataset.max_length=400 dataset.d_output=2 dataset.train_len=1210  >> $output_file
    
done

echo "All experiments completed."
