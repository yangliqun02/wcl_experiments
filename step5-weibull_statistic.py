import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import weibull_min
import os


# 输入地址
csv_path = './wcl_verification/data_with_mad_ratio.csv'
#输出地址
output_dir = './wcl_verification'

# 设置中文字体和负号显示正常（可选）
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

def fit_weibull_and_plot(data, optimizer_name, output_dir='./paper_related/output'):
    # 拟合威布尔分布
    shape, loc, scale = weibull_min.fit(data, floc=0)  # 固定位置参数为0

    # 创建输出目录
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # 绘图
    plt.figure(figsize=(10, 6))
    plt.hist(data, bins=10, density=True, alpha=0.6, color='skyblue', label='Histogram')

    # 拟合曲线
    x = np.linspace(min(data), max(data), 1000)
    pdf = weibull_min.pdf(x, shape, loc=loc, scale=scale)
    plt.plot(x, pdf, 'r-', lw=2, label=f'Weibull Fit: shape={shape:.2f}, scale={scale:.2f}')

    # plt.title(f'Relative Range for {optimizer_name}',fontsize = 30)
    plt.xlabel('Robustness Coefficient',fontsize = 20,labelpad=10)
    plt.ylabel('Density',fontsize = 20,labelpad=10)
    plt.tick_params(axis='x', labelsize=20) # x轴刻度标签字体大小
    plt.tick_params(axis='y', labelsize=20) # y轴刻度标签字体大小
    plt.legend(fontsize = 20)

    plt.grid(True)

    # 保存图像
    filename = os.path.join(output_dir, f'weibull_fit_{optimizer_name}')
    plt.savefig(filename + '.png', dpi=300, bbox_inches='tight')
    plt.savefig(filename + '.pdf', bbox_inches='tight')
    plt.close()

    # 打印参数
    print(f"Optimizer: {optimizer_name}")
    print(f"  Weibull Fit Parameters - Shape: {shape:.4f}, Scale: {scale:.4f}\n")

def analyze_mad_ratio(csv_file):
    # 读取CSV文件
    df = pd.read_csv(csv_file)
    # 检查必要列是否存在
    required_columns = ['model name', 'dataset name', 'optimizer name', 'mad_ratio']
    if not all(col in df.columns for col in required_columns):
        raise ValueError("CSV 文件必须包含 model name, dataset name, optimizer name, mad_ratio 列")

    # 提取所有样本的 mad_ratio 用于总体分析
    all_data = df['mad_ratio'].dropna().values

    # 总体拟合与绘图
    fit_weibull_and_plot(all_data, 'all optimizers',output_dir=output_dir)

    # 按 optimizer name 分组
    grouped = df.groupby('optimizer name')['mad_ratio']

    # 遍历每个 optimizer 并分析
    for optimizer_name, group_data in grouped:
        data = group_data.dropna().values
        if len(data) > 0:
            fit_weibull_and_plot(data, optimizer_name,output_dir=output_dir)

if __name__ == '__main__':
    # 输入你的CSV路径
    directory = os.path.dirname(output_dir)
    if not os.path.exists(directory):
        os.makedirs(directory)    
    analyze_mad_ratio(csv_path)