# Flow Matching with Vision Transformers

This repository provides a PyTorch implementation of image generation using **Flow Matching (Rectified Flow)** with a **Vision Transformer (DiT)** backbone.

## Features

- **DiT Architecture**: Implements the Diffusion Transformer (DiT) architecture with AdaLN-Zero conditioning blocks.
- **Rectified Flow**: Includes the core logic for Flow Matching, such as Logit-Normal timestep sampling and Heun's 2nd order ODE solver for sampling.
- **Classifier-Free Guidance**: Supports CFG for improved generation quality.
- **Modular Design**: Clean separation of model components (`image_models.py`), algorithm logic (`rectified_flow.py`), and training utilities (`trainer.py`).

## Demos

The MNIST dataset is used for training. **Figure 1** shows the original images in the dataset. **Figure 2** is the process for adding noise. **Figure 3** shows the generated images.

**Figure 1**
![Figure 1](figure_1.png)

**Figure 2**
![Figure 2](figure_2.png)

**Figure 3**
![Figure 3](figure_3.png)

## Usage

### 1. Training

The `Trainer` class in `trainer.py` manages the training process. You can define your model, optimizer, and dataloader, then start training.


```python
from trainer import Trainer
from flow_matching_model import FlowMatchingModel

model = FlowMatchingModel(...)
optimizer = torch.optim.AdamW(model.parameters())
trainer = Trainer(model, optimizer, criterion=loss_fn)

trainer.train(dataloader)
```

### 2. Generation

Use the `generate` function from `rectified_flow.py` to sample images.

```python
from rectified_flow import generate

images = generate(model, labels, cfg_scale=3.0, num_steps=50)
```
