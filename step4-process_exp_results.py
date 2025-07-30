import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import seaborn as sns
import os


# 输入地址
file_path = './wcl_verification/processed_output.csv' 
# 输出地址
output_path = './wcl_verification'

# 获取当前工作目录
current_dir = os.getcwd()

# 读取CSV文件

df = pd.read_csv(file_path)

# 计算 mad / median 的比例
df['mad_ratio'] =  df['mad']/df['median']

# 输出统计信息并保存到文本文件
stats = df['mad_ratio'].describe()
stats_file_path = os.path.join(current_dir, 'mad_ratio_stats.txt')
with open(stats_file_path, 'w') as f:
    f.write("MAD Ratio Statistics:\n")
    f.write(str(stats))

print("MAD Ratio Statistics 已保存至：", stats_file_path)

# 设置绘图风格
sns.set(style="whitegrid")

# 条形图 - 每个 model-dataset 组合的 mad ratio
plt.figure(figsize=(12, 6))
sns.barplot(x=df.index, y='mad_ratio', data=df, palette="viridis")
plt.title('MAD / Median Ratio for Each Model-Dataset Combination')
plt.xlabel('Model-Dataset Index')
plt.ylabel('Robustness Coefficient')
plt.xticks(rotation=45)
plt.tight_layout()
bar_plot_path = os.path.join(current_dir, 'mad_ratio_barplot.pdf')
plt.savefig(bar_plot_path)
plt.close()
print("条形图已保存至：", bar_plot_path)

# 直方图 - mad ratio 分布
plt.figure(figsize=(8, 4))
plt.hist(df['mad_ratio'], bins = 10, color="skyblue")
plt.title('Distribution of Robustness Coefficient')
plt.xlabel('Robustness Coefficient')
plt.ylabel('Frequency')

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import os

# 假设 df 和 current_dir 已定义
# 直方图 - mad ratio 分布
plt.figure(figsize=(8, 4))
plt.hist(df['mad_ratio'], bins=10, color="skyblue", alpha=0.7, linewidth=0.5)

plt.title('Distribution of Robustness Coefficient')
plt.xlabel('Robustness Coefficient')
plt.ylabel('Frequency')


# 计算统计量
mean_value = df['mad_ratio'].mean()
median_value = df['mad_ratio'].median()

# 添加均值（红色虚线）和中位数（橙色虚线）的垂直线
plt.axvline(mean_value, color='red', linestyle='--', linewidth=2, label=f'Mean = {mean_value:.3f}')
plt.axvline(median_value, color='orange', linestyle='-', linewidth=2, label=f'Median = {median_value:.3f}')

# 添加文本标注（避免重叠，稍微调整位置）
plt.text(mean_value, plt.ylim()[1] * 0.85, f'  Mean', 
         color='red', fontsize=10, verticalalignment='top', horizontalalignment='left')
plt.text(median_value, plt.ylim()[1] * 0.8, f'  Median', 
         color='orange', fontsize=10, verticalalignment='top', horizontalalignment='left')

plt.legend() 

plt.tight_layout()  # 自动调整布局

hist_plot_path = os.path.join(current_dir, output_path+'/mad_ratio_distribution.pdf')
plt.savefig(hist_plot_path)
hist_plot_path = os.path.join(current_dir, output_path+'/mad_ratio_distribution.png')
plt.savefig(hist_plot_path)
plt.close()
print("直方图已保存至：", output_path)

# （可选）将带有 mad_ratio 的新数据框保存为新的 CSV 文件
output_csv_path = os.path.join(current_dir, output_path+'/data_with_mad_ratio.csv')
df.to_csv(output_csv_path, index=False)
print("带 MAD Ratio 的完整数据已保存至：", output_csv_path)