import pandas as pd
import matplotlib.pyplot as plt
import os

# 设置路径
log_dir = 'training_verification'
csv1 = os.path.join(log_dir, 'training_log.csv')
csv2 = os.path.join(log_dir, 'training_log_custom.csv')

# 输出图像保存路径
output_dir = 'comparison_plots'
os.makedirs(output_dir, exist_ok=True)

# 读取CSV文件
df1 = pd.read_csv(csv1)
df2 = pd.read_csv(csv2)

# 第一步：对每个epoch求平均
df1_avg = df1.groupby('Epoch').mean(numeric_only=True).reset_index()
df2_avg = df2.groupby('Epoch').mean(numeric_only=True).reset_index()

# 第二步：截断 df2，只保留前40个epoch的数据（即：自定义训练的前40轮）
df2_truncated = df2_avg[df2_avg['Epoch'] <= 40].copy()
df2_truncated['Epoch']+=49

# 第三步：用 df1 的前50个 epoch 补充 df2，使其从0开始
df1_pre = df1_avg[df1_avg['Epoch'] < 50].copy()
df2_complete = pd.concat([df1_pre, df2_truncated], ignore_index=True)

# 添加来源标识
df1_avg['Source'] = 'Original Training'
df2_complete['Source'] = 'Custom Training'

# 合并用于绘图
combined_loss = pd.concat([
    df1_avg[['Epoch', 'Train Loss', 'Source']],
    df2_complete[['Epoch', 'Train Loss', 'Source']]
])

combined_acc = pd.concat([
    df1_avg[['Epoch', 'Test Accuracy', 'Source']],
    df2_complete[['Epoch', 'Test Accuracy', 'Source']]
])



# 保存 df1_avg（原始训练平均数据）
df1_avg.to_csv(os.path.join(output_dir, 'df1_avg.csv'), index=False)

# 保存 df2_complete（自定义训练拼接后完整的 Epoch 数据）
df2_complete.to_csv(os.path.join(output_dir, 'df2_complete.csv'), index=False)

# 保存 combined_loss（合并后的训练损失数据）
combined_loss.to_csv(os.path.join(output_dir, 'combined_loss.csv'), index=False)

# 保存 combined_acc（合并后的测试准确率数据）
combined_acc.to_csv(os.path.join(output_dir, 'combined_acc.csv'), index=False)

print("✅ 所有中间处理数据已成功保存为CSV文件。")