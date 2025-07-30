import os
import time
import uuid
import torch
import torchvision.models as models
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch import nn, optim

#输入地址
input_path = './Data'
#输出地址
output_path = './wcl_verification/weights'

# 定义要使用的模型、数据集和优化器列表
model_names = ['resnet18', 'resnet50', 'alexnet', 'vgg16', 'squeezenet1_0', 'densenet161', 'inception_v3']
dataset_names = ['cifar10', 'mnist', 'fashion_mnist', 'stl10']
optimizers = {
    'sgd': (optim.SGD, {'momentum': 0.9}),
    'adam': (optim.Adam, {}),
    'rmsprop': (optim.RMSprop, {})
}

learning_rate = 0.001
num_epochs = 1

def get_transforms(model_name, dataset_name):
    transform_list = []
    
    if model_name == 'inception_v3':
        resize_size = 299
    else:
        resize_size = 224
    
    if dataset_name in ['cifar10', 'stl10']:
        transform_list.append(transforms.Resize((resize_size, resize_size)))
    elif dataset_name in ['mnist', 'fashionmnist']:
        transform_list.extend([
            transforms.Resize((resize_size, resize_size)),
            transforms.Grayscale(3)  # 将单通道图像转换为三通道以适应某些模型输入要求
        ])
        
    transform_list.append(transforms.ToTensor())
    
    return transforms.Compose(transform_list)

def get_dataset(dataset_name, transform):
    if dataset_name == 'cifar10':
        return datasets.CIFAR10(root=input_path, train=True, download=True, transform=transform)
    elif dataset_name == 'mnist':
        return datasets.MNIST(root=input_path, train=True, download=True, transform=transform)
    elif dataset_name == 'fashion_mnist':
        return datasets.FashionMNIST(root=input_path, train=True, download=True, transform=transform)
    elif dataset_name == 'stl10':
        return datasets.STL10(root=input_path, split='train', download=True, transform=transform)
    else:
        raise ValueError("Unsupported dataset")

def adjust_model_last_layer(model, num_classes):
    if hasattr(model, 'classifier'):
        if isinstance(model.classifier, nn.Sequential):
            last_layer = list(model.classifier.children())[-1]
            if isinstance(last_layer, nn.Linear):
                last_layer.out_features = num_classes
                last_layer.weight.data.normal_(0, 0.01)
                last_layer.bias.data.zero_()
        elif isinstance(model.classifier, nn.Linear):
            model.classifier = nn.Linear(model.classifier.in_features, num_classes)
    elif hasattr(model, 'fc'):
        model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model

def ensure_dir(file_path):
    directory = os.path.dirname(file_path)
    if not os.path.exists(directory):
        os.makedirs(directory)

# 获取带唯一ID的文件名
def get_unique_filename(base_path, unique_id):
    base_name, ext = os.path.splitext(base_path)
    return f"{base_name}_{unique_id}{ext}"


def training(output_path = './wcl_verification/weights',repeat = 30):
    for i in range(repeat):
        for model_name in model_names:
            for dataset_name in dataset_names:
                for opt_name, (optimizer_cls, optimizer_kwargs) in optimizers.items():
                    print(f"Training {model_name} on {dataset_name} with {opt_name}")
                    
                    # 生成唯一ID
                    unique_id = str(int(time.time()))  # 使用当前时间戳作为唯一ID
                    
                    # 加载模型并随机初始化权重
                    model = models.__dict__[model_name](pretrained=False)
                    if model_name == 'inception_v3':
                        model.aux_logits = False
                    
                    if dataset_name in ['cifar10', 'stl10']:
                        num_classes = 10
                    elif dataset_name in ['mnist', 'fashion_mnist']:
                        num_classes = 10
                    
                    model = adjust_model_last_layer(model, num_classes)
                    
                    # 创建保存初始权重的目录并保存权重
                    initial_weights_base_path = output_path+f'/{model_name}_{dataset_name}_{opt_name}/initial_weights.pth'
                    initial_weights_path = get_unique_filename(initial_weights_base_path, unique_id)
                    ensure_dir(initial_weights_path)
                    torch.save(model.state_dict(), initial_weights_path)
                    
                    # 数据准备
                    transform = get_transforms(model_name, dataset_name)
                    train_dataset = get_dataset(dataset_name, transform)
                    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True)
                    
                    # 损失函数与优化器
                    criterion = nn.CrossEntropyLoss()
                    optimizer = optimizer_cls(model.parameters(), lr=learning_rate, **optimizer_kwargs)
                    
                    # 训练模型
                    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                    model.to(device)
                    
                    for epoch in range(num_epochs):
                        model.train()
                        running_loss = 0.0
                        for inputs, labels in train_loader:
                            inputs, labels = inputs.to(device), labels.to(device)
                            
                            optimizer.zero_grad()
                            outputs = model(inputs)
                            loss = criterion(outputs, labels)
                            loss.backward()
                            optimizer.step()
                            
                            running_loss += loss.item()
                    
                    # 创建保存训练后权重的目录并保存权重
                    trained_weights_base_path = output_path + f'{model_name}-{dataset_name}-{opt_name}/trained_weights.pth'
                    trained_weights_path = get_unique_filename(trained_weights_base_path, unique_id)
                    ensure_dir(trained_weights_path)
                    torch.save(model.state_dict(), trained_weights_path)
                    print(f'Saved Trained Weights to {trained_weights_path}')
# 遍历所有组合
def main():
    training(output_path=output_path)
if __name__ == '__main__':
    main()