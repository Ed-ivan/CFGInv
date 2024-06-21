# Ave the score
import pandas as pd
import numpy as np

# 读取CSV文件

path="/home/user/jin/SPDInv/evaluation/evaluation_result_loss=0.01.csv"
df = pd.read_csv(path)

# 计算每列的平均值，忽略NaN
# axis=0 表示沿着列的方向进行操作，skipna=True 表示在计算时忽略NaN值
average_scores = df.iloc[:, 1:].mean(axis=0, skipna=True)  # 假设第一列是图像编号，不计算平均值

# 打印结果
print("Average scores for each metric (ignoring NaN values):")
print(average_scores)