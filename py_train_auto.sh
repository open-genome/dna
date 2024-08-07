#!/bin/bash
#使用方法：bash py_train_auto.sh H4
# 定义一个数组来存储所有的best_metric值
declare -a best_metric_values
# 定义变量来存储best_metric的最大值和最小值
max_best_metric=-1
min_best_metric=100
experiment='py_nt_caduceus'
dataset_name=$1 # 从命令行参数中获取数据集名称
seeds=(2222 42 43 44 45 46 47 48 49 50)

mkdir -p "./pybash"

for seed in "${seeds[@]}"; do
    echo "Running experiment with seed: $seed"
    best_metric=-1  # 初始化best_metric为最小可能值
    
    output_file="./pybash/output_${experiment}_${dataset_name}_${seed}.txt"
    
    #如果只是统计结果，注释掉下面这行
    python train_py.py experiment="hg38/${experiment}" wandb=null train.seed=$seed dataset.dataset_name=$dataset_name > "$output_file"
    
    # metric_value=$(grep -oP '(val/mcc=|\bval/f1_macro=)\K\d+(\.\d+)?' "$output_file" | sort -nr | head -n 1)
    #metric_value 保留5位小数
    metric_value=$(grep -oP '(val/mcc=|\bval/f1_macro=)\K\d+(\.\d+)?' "$output_file" | sort -nr | head -n 1)
    echo "seed $seed: $metric_value"
    # 从这里移除了 "Metric Value" 的输出
    if (( $(echo "$metric_value > $best_metric" | bc -l) )); then
        best_metric=$metric_value  # 这里把 $metric_value_float 改成了 $metric_value
    fi
    
    # 使用printf格式化输出，确保至少5位小数
    # echo "Best Metric: $(printf "%.5f" "$best_metric")"
    
    # 将best_metric添加到数组中，并更新best_metric的最大值和最小值
    best_metric_values+=("$best_metric")
    if (( $(echo "$best_metric > $max_best_metric" | bc -l) )); then
        max_best_metric=$best_metric
    fi
    if (( $(echo "$best_metric < $min_best_metric" | bc -l) )); then
        min_best_metric=$best_metric
    fi
done

# 计算所有best_metric的均值,并输出到数组
echo "best_metric_values: $(printf "%.5f " "${best_metric_values[@]}")"
result_file="./pybash/result_${dataset_name}_${experiment}.txt"
echo -e "Experiment: $experiment \n best_metric_values: $(printf "%.5f " "${best_metric_values[@]}")" > "$result_file"