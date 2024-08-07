import os
txtpath='./pybash'
# 读取文件并解析best_mcc值
def read_best_values(file_name):
    with open (os.path.join(txtpath,file_name), "r") as file:
        # 假设best_mcc值在文件的第二行
        line = file.readline().strip()
        line = file.readline().strip()
        # 从字符串中提取best_mcc值，假设它们是以空格分隔的
        values = line.split()[1:]  # 跳过 "Experiment:" 和 "best_values:" 文本
        #字符串转换为浮点数，保留5位小数
        values = [float(v) for v in values]
        return values

# 计算均值和范围差值
def calculate_mean_and_range(values):
    if not values:
        return None, None
    # 计算均值
    mean_value = sum(values) / len(values)
    
    # 计算最大值和最小值
    max_value = max(values)
    min_value = min(values)
    
    # 计算均值与最大值和最小值之间的差值
    range_max = max_value - mean_value
    range_min = mean_value - min_value
    
    return mean_value, max(range_max, range_min)

# 主程序
if __name__ == "__main__":
    # 在当前目录下，找到所有result_开头的txt文件
     #以表格的形式输出，第一列是Mean，第二列是range diff,后面是每个seed的best_mcc值 每行呆逼包一个文件
    seed = [2222,42,43,44,45,46,47,48,49,50]
    import pandas as pd
    df = pd.DataFrame(columns=["Mean", "Diff"] + [f"{seed[i]}" for i in range(len(seed))])
    
    result_files = [f for f in os.listdir(txtpath) if f.startswith("result_") and f.endswith(".txt")]
    for file_name in result_files:
        values = read_best_values(file_name)
        values = [v*100 for v in values]
        # print(file_name+":",values)
        # 计算均值和范围差值
        mean_value, range_diff = calculate_mean_and_range(values)
        
        # 输出结果
        if mean_value is not None:
            # print(f"Mean: {mean_value:.5f}")
            # print(f"Range Difference (max/min - mean): {range_diff:.5f}")
            # df.loc[file_name] = [mean_value, range_diff] + values
            #保留两位小数
            df.loc[file_name] = [round(mean_value,5), round(range_diff,5)] + [round(v,5) for v in values]
        else:
            print("No valid values found in the file.")
            df.loc[file_name] = [None, None] + values
        # print("---------------")

    #输出2位小数
    # pd.set_option('precision', 2)
    print(df)
    df.to_csv("result_bash.csv")