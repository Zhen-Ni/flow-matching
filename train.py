#!/usr/bin/env python3

import torch
import torchvision
from torch.utils.data import DataLoader
from rectified_flow import sample_timesteps, add_noise
from flow_matching_model import FlowMatchingModel
from trainer import Trainer


def preprocess(x: torch.Tensor, y: torch.Tensor):
    """Preprocess function for Flow Matching training.
    
    Converts raw images and labels into model inputs and targets.
    
    Args:
        x: Clean images [B, C, H, W].
        y: Class labels [B].
        
    Returns:
        Tuple[Tuple[torch.Tensor, ...], torch.Tensor]:
            - Input tuple for the model: (x_t, t, y)
            - Target velocity: v_target
    """
    # 1. Sample timesteps
    batch_size = x.shape[0]
    t = sample_timesteps(batch_size, device=x.device)
    
    # 2. Add noise to get x_t and target velocity v_t
    # add_noise returns (x_t, v_target)
    x_t, v_target = add_noise(x, t)
    
    # 3. Construct model inputs
    # The model forward signature is forward(x, t, y)
    # We pack them into a tuple. The Trainer's _forward method
    # handles unpacking tuples: self.model(*x)
    model_input = (x_t, t, y)
    
    return model_input, v_target


if __name__ == '__main__':
    model_name = 'best.th'

    # 定义数据变换：
    # 1. ToTensor(): 将图像转为 Tensor，数值范围 [0, 1]
    # 2. Normalize(): 执行 (x - 0.5) / 0.5，将数值范围映射到 [-1, 1]
    # Flow Matching 假设先验分布为标准高斯分布 N(0, 1)，
    # 将数据归一化到 [-1, 1] 可以使其与噪声分布的尺度对齐，有助于训练收敛。
    transform = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize((0.5,), (0.5,))
    ])

    # 直接使用 torchvision 的 MNIST 数据集
    dataset_train = torchvision.datasets.MNIST(
        root='../.cache',
        train=True,
        transform=transform,
        download=True
    )
    dataset_validate = torchvision.datasets.MNIST(
        root='../.cache',
        train=False,
        transform=transform,
        download=True
    )

    loader = DataLoader(dataset_train, batch_size=256, shuffle=True)
    loader_val = DataLoader(dataset_validate, batch_size=256)

    epochs = 100
    best_loss: float | None
    try:
        trainer = Trainer.load('trainer.trainer')
        best_loss = min(trainer.history['validate_loss'])
        model = trainer.model
        print("Use stored trainer, continue training.")
    except FileNotFoundError:
        model = FlowMatchingModel(1, 28, 28, 4, 128)
        lr = 1e-3
        optimizer = torch.optim.AdamW(model.parameters(), lr)
        criterion = torch.nn.MSELoss()
        trainer = Trainer(model, optimizer, criterion)
        trainer.device = 'xpu'
        best_loss = None
        print('New trainer generated.')

        print(f'current/total epochs [{trainer.epoch}/{epochs}]')
    print('trainer device:', trainer.device)
    
    while trainer.epoch < epochs:
        trainer.train(loader, preprocess=preprocess)
        trainer.validate(loader_val, preprocess=preprocess)
        trainer.save(device='cpu')
        
        # Save the best model.
        current_loss = trainer.history['validate_loss'][-1]
        if (best_loss is None) or (current_loss < best_loss):
            print('This model will be saved as the best model.')
            best_loss = current_loss
            with open(model_name, 'wb') as f:
                torch.save(trainer.model, f)

    print('Training process finished.\n')
