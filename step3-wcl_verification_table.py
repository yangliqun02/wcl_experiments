import os
import pandas as pd
import numpy as np

#输入地址
csv_folder_path = './wcl_verification/distance'
#输出地址
output_folder_path = './wcl_verification'

def calculate_median_stats_for_dir(path, output_file):
    results = []

    for filename in os.listdir(path):
        if filename.endswith('.csv'):
            file_path = os.path.join(path, filename)
            try:
                df = pd.read_csv(file_path, index_col=0)  # 跳过原始索引列
                numeric_df = df.select_dtypes(include=[np.number])  # 只保留数值列
                data = numeric_df.values.flatten()

                median_value = np.median(data)
                mad = np.median(np.abs(data - median_value))

                mean_value = np.mean(data)
                std_value = np.std(data)

                results.append({
                    'filename': filename,
                    'median': median_value,
                    'mad': mad,
                    'mean_value':mean_value,
                    'std_value':std_value
                })

            except Exception as e:
                print(f"无法处理文件 {filename}：{e}")

    # 构建 DataFrame 并设置 filename 为 index
    result_df = pd.DataFrame(results).set_index('filename')

    # 保存为 CSV 文件
    result_df.to_csv(output_file)
    print(f"结果已保存至 {output_file}")


import pandas as pd

def reproduce_table(file_path = './result.csv'):
    df = pd.read_csv(file_path)
    df.columns = ['filename','median','mad','mean','std']
    # 提取 model name, dataset name, optimizer name
    split_df = df['filename'].str.extract(r'([^-]+)-([^-]+)-([^-]+)\.csv')
    # 将提取出的三列加入原数据框
    df[['model name', 'dataset name', 'optimizer name']] = split_df
    df = df.drop(columns=['filename'])
    # 输出处理后的DataFrame
    print(df)
    # 如果需要保存为新的CSV文件
    df.to_csv(output_folder_path+'/processed_output.csv', index=False)


# 示例调用：
if __name__ == '__main__':
    
    directory = os.path.dirname(csv_folder_path)
    if not os.path.exists(directory):
        os.makedirs(directory)
    reform_data_path=output_folder_path+'/result.csv'
    calculate_median_stats_for_dir(csv_folder_path,output_file=reform_data_path) 
    reproduce_table(file_path=reform_data_path)