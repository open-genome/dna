#!/bin/bash

# 定义一个数组来存储所有的best_mcc值
declare -a best_mcc_values
# 定义变量来存储best_mcc的最大值和最小值
max_best_mcc=-1
min_best_mcc=100
# experiment='py_nt_denoisecnn'
experiment='py_nt_caduceus'
dataset_name='H4'
# 随机种子列表
seeds=(2222 42 43 44 45 46 47 48 49 50)

# 确保输出目录存在
mkdir -p "./pybash"

# 对于每个随机种子，执行训练并提取val/mcc的最大值
# best_mcc_values=(1 2 3 4 5 6)
for seed in "${seeds[@]}"; do
    echo "Running experiment with seed: $seed"
    best_mcc=-1  # 初始化best_mcc为最小可能值
    
    # 执行训练命令，并重定向输出到文件
    output_file="./pybash/output_${dataset_name}_${seed}.txt"
    python train_py.py experiment="hg38/${experiment}" wandb=null train.seed=$seed dataset.dataset_name=$dataset_name > "$output_file"
    
    # 从输出文件中提取val/mcc的值
    mcc_value=$(grep -oP 'val/mcc=\K\d+\.\d+' "$output_file" | sort -nr | head -n 1)
    
    # 更新best_mcc
    if (( $(echo "$mcc_value > $best_mcc" | bc -l) )); then
        best_mcc=$mcc_value
    fi
    
    echo "Best MCC: $best_mcc"
    
    # 将best_mcc添加到数组中，并更新best_mcc的最大值和最小值
    best_mcc_values+=("$best_mcc")
    if (( $(echo "$best_mcc > $max_best_mcc" | bc -l) )); then
        max_best_mcc=$best_mcc
    fi
    if (( $(echo "$best_mcc < $min_best_mcc" | bc -l) )); then
        min_best_mcc=$best_mcc
    fi
done

# # 计算所有best_mcc的均值,并输出到数组
echo best_mcc_values: ${best_mcc_values[@]}
result_file="./pybash/result_${dataset_name}_${experiment}.txt"
echo -e "Experiment: $experiment \n best_mcc_values: ${best_mcc_values[@]}" > "$result_file"
# mean=$(echo "${best_mcc_values[@]}" | awk '{ sum += $1; n++ } END { print sum/n; }')
# echo mean: $mean
# # 计算均值与best_mcc中最大值和最小值之间的差值，取较大者
# range_max=$(echo "$max_best_mcc - $mean" | bc -l)
# range_min=$(echo "$mean - $min_best_mcc" | bc -l)
# range_diff=$(echo "$range_max > $range_min ? $range_max : $range_min" | bc -l)

# # 输出结果到屏幕
# echo -e "Mean(best_mcc)\tRange Difference\tRandom Seeds"
# echo -e "$mean\t$range_diff\t${seeds[*]}"

# # 把对应的结果输出到文件
# result_file="./pybash/result_${dataset_name}.txt"
# echo -e "Experiment: $experiment" > "$result_file"
# echo -e "Mean(best_mcc)\tRange Difference" >> "$result_file"
# echo -e "$mean\t$range_diff" >> "$result_file"
# # 直接输出整个best_mcc_values数组
# for value in "${best_mcc_values[@]}"; do
#     echo "Seed Value: $value" >> "$result_file"
# done