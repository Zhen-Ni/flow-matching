#!/usr/bin/env python3

from __future__ import annotations
import os
import matplotlib.pyplot as plt
import torch
from rectified_flow import add_noise, generate

plt.rc('font', family='STIXGeneral', weight='normal', size=10)
plt.rc('mathtext', fontset='stix')


def show_mnist_batch(dataset, n_rows=2, n_cols=5):
    fig, axes = plt.subplots(
        n_rows, n_cols, figsize=(10, 4)
    )
    axes = axes.flatten()

    for i, ax in enumerate(axes):
        image, label = dataset[i]
        image = image.squeeze()
        ax.imshow(image, cmap="gray")
        ax.set_title(f"Label: {label}")
        ax.axis("off")

    plt.tight_layout()


def show_forward(dataset, n_cols=5, n_rows=2):
    """Show images at different flow matching steps.

    Parameters
    ----------
    dataset : Dataset
        Dataset to get images from.
    n_cols : int, optional
        Number of columns in the plot. Default is 5.
    n_rows : int, optional
        Number of rows in the plot. Default is 2.
    """
    fig, axes = plt.subplots(
        n_rows, n_cols, figsize=(10, 2 * n_rows)
    )

    # Select different timesteps evenly spaced
    timesteps = torch.linspace(
        1,
        0,
        n_cols,
    )

    for row in range(n_rows):
        # Get a sample image
        image, label = dataset[row]
        # Add batch dimension and move to device
        x_1 = image.unsqueeze(0).to(device)
        x_t = add_noise(x_1, timesteps)[0]
        
        for i, ax in enumerate(axes[row]):
            t = timesteps[i]
            noisy_image = x_t[i].squeeze().cpu().numpy()
            ax.imshow(noisy_image, cmap="gray")
            ax.set_title(f"t = {t.item()}")
            ax.axis("off")

    plt.tight_layout()


def show_generation(
        model: torch.nn.Module,
        labels: list[int],
        cfg_scale: float = 3.0,
        num_steps: int = 20,
        img_size: tuple[int, int, int] = (1, 28, 28),
        n_cols: int | None = None,
):
    """Generate and display images for specified labels using DDPM.

    Args:
        generator: DitGenerator instance.
        labels: List of labels to generate.
        cfg_scale: CFG scale.
        num_steps: Number of steps for generation.
        img_size: Image size.
        n_cols: Number of columns in the plot.
    """
    n_images = len(labels)

    if n_cols is None:
        n_cols = n_images

    n_rows = (n_images + n_cols - 1) // n_cols

    fig, axes = plt.subplots(
        n_rows, n_cols, figsize=(n_cols * 2, n_rows * 2)
    )

    # Flatten axes for easy iteration
    axes = axes.flatten()

    device=next(iter(model.parameters())).device
    imgs = generate(model,
                    y=torch.tensor(labels).to(device),
                    x0=torch.randn([n_images]+list(img_size)),
                    cfg_scale=cfg_scale,
                    num_steps=num_steps
                    )
                    
    
    for i, label in enumerate(labels):

        img = imgs[i].cpu().squeeze().numpy()

        axes[i].imshow(img, cmap="gray")

        # Set title: "Uncond" for label 10
        title = f"Label: {label}"
        if label == 10:
            title = "Label: Uncond"
        axes[i].set_title(title)
        axes[i].axis("off")

    # Hide any unused subplots
    for i in range(n_images, len(axes)):
        axes[i].axis("off")

    plt.tight_layout()



if __name__ == '__main__':
    import torchvision

    # Setup device
    device = torch.device('cpu')

    dataset = torchvision.datasets.MNIST(
        root='../.cache',
        train=True,
        transform=torchvision.transforms.ToTensor(),
        download=False
    )

    show_mnist_batch(dataset, n_rows=2, n_cols=5)

    show_forward(dataset, n_cols=5, n_rows=3)

    model_path = 'best.th'
    if os.path.exists(model_path):
        # Load the full model object directly
        # weights_only=False is required because train_diffusion.py
        # saves the whole model object via torch.save(model, f)
        model = torch.load(
            model_path,
            map_location=device,
            weights_only=False
        )
        model.eval()
        print(f"Model loaded from {model_path}")

        # Generate labels 0-9 and two unconditional (label 10)
        labels = list(range(10)) + [10, 10]
        
        print("Showing flow matching generation...")
        show_generation(
            model,
            labels=labels,
            n_cols=4,
        )

    else:
        print(f"Error: Model file '{model_path}' not found.")
        print("Skipping generation visualization.")

    model_path = 'reflow_best.th'
    if os.path.exists(model_path):
        model = torch.load(
            model_path,
            map_location=device,
            weights_only=False
        )
        model.eval()
        print(f"Model loaded from {model_path}")

        # Generate labels 0-9 and two unconditional (label 10)
        labels = list(range(10)) + [10, 10]
        
        print("Showing reflow generation...")
        show_generation(
            model,
            labels=labels,
            n_cols=4,
            cfg_scale=None,
            num_steps=5
        )

    else:
        print(f"Error: Model file '{model_path}' not found.")
        print("Skipping generation visualization.")

    plt.show()
