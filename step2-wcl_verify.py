import os
import torch
import pandas as pd

# 输入地址
root_dir = './wcl_verification/weights'
# 输出地址
output_dir = './wcl_verification/distance'

# 定义要使用的模型、数据集和优化器列表
model_names = ['resnet18', 'resnet50', 'alexnet', 'vgg16', 'squeezenet1_0', 'densenet161', 'inception_v3']
dataset_names = ['cifar10', 'mnist', 'fashion_mnist', 'stl10']
optimizers = ['sgd', 'adam', 'rmsprop']  # 只需要名称即可

def get_device():
    """返回可用的最佳设备（GPU 或 CPU）"""
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def calculate_euclidean_distance(tensor1, tensor2,device):
    """计算两个张量之间的欧式距离"""
    # tensor1.to(get_device())
    # tensor2.to(get_device())
    if not (tensor1.is_floating_point() and tensor2.is_floating_point()):
        return 0.0
    return torch.norm(tensor1 - tensor2, p=2).item()

def process_model_dataset_optimizer(model_name, dataset_name, optimizer_name,device):
    folder_name = f"{model_name}_{dataset_name}_{optimizer_name}"
    folder_path = os.path.join(root_dir, folder_name)

    if not os.path.exists(folder_path):
        print(f"Folder {folder_path} does not exist. Skipping.")
        return []

    files = os.listdir(folder_path)
    
    # 提取初始和训练后的权重文件
    initial_files = [f for f in files if f.startswith('initial_weights_') and f.endswith('.pth')]
    trained_files = [f for f in files if f.startswith('trained_weights_') and f.endswith('.pth')]

    results = []

    # 遍历每个 initial 文件，尝试匹配对应的 trained 文件
    for init_file in initial_files:
        timestep = init_file.split('_')[2]
        match_trained = [f for f in trained_files if f'trained_weights_{timestep}' in f]

        if not match_trained:
            print(f"No trained weights found for timestep {timestep}. Skipping.")
            # print(len(trained_files))
            # print(trained_files)
            # print(f'trained_weights_{timestep}')
            continue

        trained_file = match_trained[0]

        print(f"Processing: {folder_name}, Timestep: {timestep}")

        # 加载权重
        init_path = os.path.join(folder_path, init_file)
        trained_path = os.path.join(folder_path, trained_file)

        state_dict_initial = torch.load(init_path,map_location=device)
        state_dict_trained = torch.load(trained_path,map_location=device)

        total_distance = 0.0

        # 对每一层计算权重差异
        for key in state_dict_initial:
            if key in state_dict_trained:
                w1 = state_dict_initial[key]
                w2 = state_dict_trained[key]
                distance = calculate_euclidean_distance(w1, w2, device)
                total_distance += distance
        # 记录结果
        results.append({
            'Timestep': timestep,
            'Total Euclidean Distance': total_distance
        })

    return results

def main():
    device = get_device()
    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)

    # 遍历所有组合
    for model_name in model_names:
        for dataset_name in dataset_names:
            for optimizer_name in optimizers:
                print(f"\nProcessing: {model_name} - {dataset_name} - {optimizer_name}")
                results = process_model_dataset_optimizer(model_name, dataset_name, optimizer_name,device)

                if results:
                    df = pd.DataFrame(results)
                    output_csv = os.path.join(output_dir, f"{model_name}-{dataset_name}-{optimizer_name}.csv")
                    df.to_csv(output_csv, index=False)
                    print(f"Results saved to {output_csv}")

if __name__ == "__main__":
    main()