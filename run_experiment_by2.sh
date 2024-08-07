seeds=(2222 42 43 44 45 46 47 48 49 50)

# Output file
output_file="experiment_output.txt"

# Loop over seeds
for seed in "${seeds[@]}"
do
    echo "Running experiment with seed $seed"
    echo "Running experiment with seed $seed" >> $output_file
   CUDA_VISIBLE_DEVICES=1,3  /mnt/nas/share2/home/by/envs/hndna/bin/python train.py experiment='hg38/nt_ablation' wandb=null train.seed=$seed dataset.dataset_name='H4' >> $output_file
done

echo "All experiments completed."
