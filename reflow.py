#!/usr/bin/env python3

import tqdm

import torch
from torch.utils.data import Dataset, DataLoader
from rectified_flow import generate, sample_timesteps
from flow_matching_model import FlowMatchingModel
from trainer import Trainer

import train

DEVICE = 'xpu'


def generate_reflow_dataset(teacher: torch.nn.Module,
                            dataset_size: int,
                            batch_size: int):
    """Generate dataset for reflow training.

    This function generates pairs of noise and corresponding images
    using a pre-trained teacher model. It samples random noise (x0)
    and class labels (y), then uses the teacher model to generate
    images (x1) via the rectified flow generation process. The
    resulting (x0, x1) pairs can be used to train a student model
    for reflow.

    Args:
        teacher: The pre-trained teacher model used to generate images.
        dataset_size: The total number of data pairs to generate.
        batch_size: The batch size used during the generation process.

    Returns:
        A tuple containing three lists:
            - x0_list: List of noise tensors.
            - x1_list: List of generated image tensors.
            - y_list: List of class label tensors.
    """
    teacher.eval()
    device = next(iter(teacher.parameters())).device
    x0_list = []                # Noise
    x1_list = []                # Image
    y_list = []                 # Label
    n_loops = (dataset_size - 1) // batch_size + 1
    tq = tqdm.tqdm(range(n_loops),
                   desc="generate",
                   ncols=None,
                   leave=False,
                   unit="batch")
    for i in tq:
        batch_size_i = (dataset_size % batch_size if
                        i+1==n_loops else batch_size)
        x0 = torch.randn([batch_size_i, 1, 28, 28],
                         device=device) # Size of MNIST dataset
        y = torch.randint(0, 10, (batch_size_i,), device=device)
        x1 = generate(teacher, x0, y, cfg_scale=3)
        x0_list.extend(x0.cpu().detach())
        x1_list.extend(x1.cpu().detach())
        y_list.extend(y.cpu().detach())
    return x0_list, x1_list, y_list


class ReflowDataset(Dataset):
    """Dataset for reflow training containing noise, image, and label triplets.

    This dataset stores the generated (x0, x1, y) triplets used for
    training a student model in the reflow process. It provides
    access to the initial noise, the generated image, and the
    corresponding class label.

    Attributes:
        x0: Stacked noise tensors.
        x1: Stacked generated image tensors.
        y: Stacked class label tensors.
    """
    def __init__(self, x0_list, x1_list, y_list):
        self.x0 = torch.stack(x0_list)
        self.x1 = torch.stack(x1_list)
        self.y = torch.stack(y_list)

    def __len__(self):
        return len(self.x0)

    def __getitem__(self, idx):
        return (self.x0[idx], self.x1[idx]), self.y[idx]


def preprocess(x, y):
    """Preprocess function for Reflow training.
    
    Converts (x0, x1) pairs and labels into model inputs and targets.
    
    Args:
        x: Tuple of tensors (x0, x1) representing noise and generated images.
        y: Class labels [B].
        
    Returns:
        Tuple[Tuple[torch.Tensor, ...], torch.Tensor]:
            - Input tuple for the model: (x_t, t, y)
            - Target velocity: v
    """
    x0, x1 = x
    batch_size = y.shape[0]
    t = sample_timesteps(batch_size, device=x0.device)
    t_reshape = t.view(-1, 1, 1, 1)
    
    x_t = x0 * (1 - t_reshape) + x1 * t_reshape
    v = -x0 + x1
    
    # 3. Construct model inputs
    # The model forward signature is forward(x, t, y)
    # We pack them into a tuple. The Trainer's _forward method
    # handles unpacking tuples: self.model(*x)
    model_input = (x_t, t, y)
    
    return model_input, v
    


if __name__ == '__main__':
    import os
    
    device = 'xpu'
    model_path = 'best.th'
    reflow_model_path = 'reflow_best.th'
    trainer_filename = 'reflow1.trainer'
    teacher = None
    if os.path.exists(model_path):
        teacher = torch.load(
            model_path,
            map_location=device,
            weights_only=False
        )
        teacher.eval()
        print(f"Teacher model loaded from {model_path}")

    # Generate training dataset
    print("Generating training dataset...")
    x0, x1, y = generate_reflow_dataset(teacher, 24576, 512)
    dataset = ReflowDataset(x0, x1, y)
    
    loader = DataLoader(dataset, batch_size=256, shuffle=True)

    # Generate validation dataset
    # We use a smaller size for validation to save time
    print("Generating validation dataset...")
    x0_val, x1_val, y_val = generate_reflow_dataset(teacher, 4096, 512)
    dataset_val = ReflowDataset(x0_val, x1_val, y_val)
    loader_val = DataLoader(dataset_val, batch_size=256, shuffle=False)

    epochs = 100
    best_loss: float | None
    try:
        trainer = Trainer.load(trainer_filename)
        best_loss = min(trainer.history['validate_loss'])
        model = trainer.model
        print("Use stored trainer, continue training.")
    except FileNotFoundError:
        model = FlowMatchingModel(1, 28, 28, 4, 128,
                                  class_dropout_prob=0.)
        # Copy parameters from the teacher model
        if teacher is not None:
            model.load_state_dict(teacher.state_dict())
        lr = 1e-3
        optimizer = torch.optim.AdamW(model.parameters(), lr)
        criterion = torch.nn.MSELoss()
        trainer = Trainer(model, optimizer, criterion,
                          filename=trainer_filename)
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
            with open(reflow_model_path, 'wb') as f:
                torch.save(trainer.model, f)

    print('Training process finished.\n')
