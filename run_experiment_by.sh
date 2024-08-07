# #!/bin/bash

# # Seeds array
# seeds=(2222 42 43 44 45 46 47 48 49 50)

# # Output file
# output_file="experiment_output.txt"

# # Loop over seeds
# for seed in "${seeds[@]}"
# do
#     echo "Running experiment with seed $seed"
#     echo "Running experiment with seed $seed" >> $output_file
#     /mnt/nas/share2/home/by/envs/hndna/bin/python train.py experiment='hg38/nt_ablation' wandb=null train.seed=$seed >> $output_file
# done

# echo "All experiments completed."

#!/bin/bash

# 定义一个数组，包含所有的数据集名称
datasets=("enhancer" "enhancer_types" "H3" "H3K4me1" "H3K4me2" "H3K4me3" "H3K9ac" "H3K14ac" "H3K36me3" "H3K79me3" "H4" "H4ac" "promoter_all" "promoter_non_tata" "promoter_tata" "splice_sites_acceptor" "splice_sites_donor")

# 循环遍历数组中的每个元素
for dataset in ${datasets[@]}; do
    # 打印当前正在处理的数据集名称
    echo "Processing ${dataset}..."

    # 运行命令，并将输出重定向到以数据集名称命名的.txt文件中
    nohup /mnt/nas/share2/home/by/envs/hndna/bin/python train.py experiment='hg38/hg38_bert' wandb=null dataset.dataset_name="${dataset}" train.seed=48 trainer.devices=4  dataset.batch_size=128 > "${dataset}.txt" &

    # 等待当前命令完成
    wait
done

echo "All datasets processed."