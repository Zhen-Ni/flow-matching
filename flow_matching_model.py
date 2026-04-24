"""
Transformer-based Flow Matching Model for Image Generation.
"""

import torch
import torch.nn as nn

from image_models import (
    modulate,
    PatchEmbedder,
    PatchUnembedder,
    PositionalEmbedder,
    Mlp,
    AdaLNZeroBlock,
)


class TimestepEmbedder(nn.Module):
    """Embeds scalar timesteps into vector representations.

    Uses sinusoidal positional encoding scheme to map diffusion
    timesteps to high-dimensional vectors, then processes through
    an MLP to obtain final timestep embeddings.
    """

    def __init__(
        self,
        embed_dim: int,
        hidden_size: int
    ):
        """Initialize the TimestepEmbedder.

        Args:
            embed_dim: Dimension of sinusoidal encoding.
            hidden_size: Hidden dimension of the MLP.
        """
        super().__init__()
        self.embed_dim = embed_dim

        half_dim = embed_dim // 2
        freqs = torch.exp(
            torch.arange(0, half_dim).float()
            * (-torch.log(torch.tensor(10000.0)) / embed_dim)
        )
        # Use register_buffer so freqs moves with the model .to(device)
        self.register_buffer('freqs', freqs)

        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """Embed timesteps into vector space.

        Args:
            t: Timestep tensor of shape [batch_size], normalized to [0, 1].

        Returns:
            Embedded timestep tensor of shape [batch_size, hidden_size].
        """
        # Scale t from [0, 1] to [0, 1000] to utilize the full capacity
        # of the sinusoidal encoding. Without this scaling, the input
        # to sin/cos would be too small, resulting in near-linear encoding.
        t = t * 1000.0

        args = t[:, None] * self.freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if self.embed_dim % 2:
            embedding = torch.cat(
                [embedding, torch.zeros(embedding.shape[0], 1)],
                dim=-1
            )
        t_emb = self.mlp(embedding)
        return t_emb


class LabelEmbedder(nn.Module):
    """Embeds class labels into vector representations.

    Also supports label dropout mechanism for classifier-free
    guidance. During training, some labels are randomly replaced
    with a blank label to enable unconditional generation.

    Attributes:
        num_classes: Number of classes.
        class_dropout_prob: Probability of label dropout.
    """

    def __init__(
        self,
        num_classes: int,
        hidden_size: int,
        class_dropout_prob: float
    ):
        """Initialize the LabelEmbedder.

        Args:
            num_classes: Number of classes.
            hidden_size: Dimension of embedding vectors.
            class_dropout_prob: Probability of label dropout.
        """
        super().__init__()
        self.embedding_table = nn.Embedding(num_classes + 1, hidden_size)
        self.num_classes = num_classes
        self.class_dropout_prob = class_dropout_prob

    def token_drop(
        self,
        labels: torch.Tensor,
        force_drop_ids: torch.Tensor | None = None
    ) -> torch.Tensor:
        """Drop labels to enable classifier-free guidance.

        Replaces labels with blank label (index num_classes) with
        a certain probability during training, enabling the model
        to learn both conditional and unconditional generation.

        Args:
            labels: Original labels tensor of shape [batch_size].
            force_drop_ids: Optional forced drop indices.

        Returns:
            Processed labels tensor of shape [batch_size].
        """
        if force_drop_ids is None:
            drop_ids = (
                torch.rand(labels.shape[0], device=labels.device)
                < self.class_dropout_prob
            )
        else:
            drop_ids = force_drop_ids == 1
        labels = torch.where(drop_ids, self.num_classes, labels)
        return labels

    def forward(
        self,
        labels: torch.Tensor,
        train: bool,
        force_drop_ids: torch.Tensor | None = None
    ) -> torch.Tensor:
        """Embed labels into vector space.

        Args:
            labels: Labels tensor of shape [batch_size].
            train: Whether in training mode (enables label dropout).
            force_drop_ids: Optional forced drop indices.

        Returns:
            Embedded labels tensor of shape [batch_size, hidden_size].
        """
        if train or (force_drop_ids is not None):
            labels = self.token_drop(labels, force_drop_ids)
        embeddings = self.embedding_table(labels)
        return embeddings


class FinalLayer(nn.Module):
    """The final layer of the model.

    Responsible for decoding transformer outputs back to pixel space.
    Applies adaptive layer normalization (adaLN) modulation, consistent
    with AdaLNZeroBlock.
    """

    def __init__(
        self,
        hidden_size: int,
    ):
        """Initialize the FinalLayer.

        Args:
            hidden_size: Hidden dimension.
        """
        super().__init__()
        self.norm_final = nn.LayerNorm(
            hidden_size,
            elementwise_affine=False,
            eps=1e-6
        )
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 2 * hidden_size, bias=True)
        )

    def forward(
        self,
        x: torch.Tensor,
        c: torch.Tensor
    ) -> torch.Tensor:
        """Compute the final layer output.

        Args:
            x: Input tensor of shape [batch_size, seq_len, hidden_size].
            c: Conditioning tensor of shape [batch_size, hidden_size].

        Returns:
            Output tensor of shape [batch_size, seq_len, hidden_size].
        """
        shift, scale = self.adaLN_modulation(c).chunk(2, dim=1)
        x = modulate(self.norm_final(x), shift, scale)
        return x


class FlowMatchingModel(nn.Module):
    """Flow Matching model with a Transformer backbone.

    This model implements Flow Matching (Rectified Flow) using a
    Vision Transformer architecture. It takes images at a specific
    timestep, timesteps, and class labels as input and predicts
    the velocity field that transports noise to data.

    Architecture:
        1. PatchEmbedder: Divides input image into non-overlapping
           patches and embeds them.
        2. PositionalEmbedder: Adds positional encoding.
        3. TimestepEmbedder: Embeds timesteps into vector space.
        4. LabelEmbedder: Embeds class labels into vector space.
        5. AdaLNZeroBlocks: Stacked transformer blocks with adaptive
           layer normalization zero conditioning.
        6. FinalLayer: Final normalization and modulation layer.
        7. PatchUnembedder: Decodes patch sequence back to image.

    Example:
        >>> model = FlowMatchingModel(
        ...     img_channel=1,
        ...     img_height=28,
        ...     img_width=28,
        ...     patch_size=4,
        ...     hidden_size=128,
        ...     depth=3,
        ...     num_heads=4
        ... )
        >>> x = torch.randn(2, 1, 28, 28)  # Noisy image
        >>> t = torch.rand(2)  # Timesteps in [0, 1]
        >>> y = torch.randint(0, 10, (2,))  # Class labels
        >>> v_pred = model(x, t, y)  # Predicted velocity
    """

    def __init__(
        self,
        img_channel: int,
        img_height: int,
        img_width: int,
        patch_size: int,
        hidden_size: int,
        depth: int = 3,
        num_heads: int = 8,
        mlp_ratio: float = 4.0,
        class_dropout_prob: float = 0.1,
        num_classes: int = 10,
    ):
        """Initialize the Flow Matching model.

        Args:
            img_channel: Number of input channels.
            img_height: Input image height.
            img_width: Input image width.
            patch_size: Patch size for dividing image into non-overlapping
                blocks.
            hidden_size: Hidden dimension of the Transformer.
            depth: Number of AdaLNZeroBlocks.
            num_heads: Number of attention heads.
            mlp_ratio: Ratio of MLP hidden dimension to hidden_size.
            class_dropout_prob: Probability of class label dropout.
            num_classes: Number of classes.
        """
        super().__init__()
        self.img_channel = img_channel
        self.img_height = img_height
        self.img_width = img_width

        self.patch_embedder = PatchEmbedder(
            img_channel, hidden_size, patch_size
        )
        self.pos_embedder = PositionalEmbedder(hidden_size)
        self.t_embedder = TimestepEmbedder(256, hidden_size)
        self.y_embedder = LabelEmbedder(
            num_classes, hidden_size, class_dropout_prob
        )

        self.blocks = nn.ModuleList(
            [AdaLNZeroBlock(hidden_size, num_heads, mlp_ratio)
             for _ in range(depth)]
        )
        self.final_layer = FinalLayer(hidden_size)
        self.patch_unembedder = PatchUnembedder(
            hidden_size, patch_size, img_channel,
            img_height, img_width
        )

    def forward(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        y: torch.Tensor
    ) -> torch.Tensor:
        """Forward pass of Flow Matching model.

        Takes image at timestep t, timesteps, and class labels as input
        and predicts the velocity field.

        Args:
            x: Input image tensor of shape [batch_size, img_channel,
                img_height, img_width].
            t: Diffusion timestep tensor of shape [batch_size].
            y: Class label tensor of shape [batch_size].

        Returns
            Predicted velocity tensor with same shape as input x.

        """
        # Step 1: Image patch embedding
        # x: [batch_size, img_channel, img_height, img_width]
        #    -> [batch_size, num_patches, hidden_size]
        # where num_patches = (img_height/patch_size) * (img_width/patch_size)
        x = self.patch_embedder(x)

        # Step 2: Add positional encoding
        # x: [batch_size, num_patches, hidden_size]
        #    -> [batch_size, num_patches, hidden_size]
        x = self.pos_embedder(x)

        # Step 3: Timestep embedding
        # t: [batch_size,] -> [batch_size, hidden_size]
        t = self.t_embedder(t)

        # Step 4: Class label embedding
        # y: [batch_size,] -> [batch_size, hidden_size]
        y = self.y_embedder(y, self.training)

        # Step 5: Fuse timestep and label as conditioning vector
        # c: [batch_size, hidden_size]
        c = t + y

        # Step 6: Process through AdaLN-Zero Blocks
        # x: [batch_size, num_patches, hidden_size]
        # c: [batch_size, hidden_size]
        # output: [batch_size, num_patches, hidden_size]
        for block in self.blocks:
            x = block(x, c)

        # Step 7: Final layer processing
        # output x: [batch_size, num_patches, hidden_size]
        x = self.final_layer(x, c)

        # Step 8: Decode patches back to image
        # x: [batch_size, num_patches, hidden_size]
        #    -> [batch_size, img_channel, img_height, img_width]
        x = self.patch_unembedder(x)
        return x

    def forward_with_cfg(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        y: torch.Tensor,
        cfg_scale: float
    ) -> torch.Tensor:
        """Forward pass with classifier-free guidance for Flow Matching.

        Args:
            x: Input image tensor of shape [batch_size, img_channel,
                img_height, img_width].
            t: Diffusion timestep tensor of shape [batch_size].
            y: Class label tensor of shape [batch_size].
            cfg_scale: Strength of classifier-free guidance.
                Value of 1.0 means no guidance.

        Returns:
            Predicted velocity tensor with CFG applied, same shape as input x.
        """
        # 1. Create unconditional labels (using num_classes as the blank label)
        y_uncond = torch.full_like(y, self.y_embedder.num_classes)

        # 2. Concatenate inputs to compute conditional and unconditional predictions simultaneously
        # x: [B, C, H, W] -> [2B, C, H, W]
        x_in = torch.cat([x, x], dim=0)
        t_in = torch.cat([t, t], dim=0)
        y_in = torch.cat([y, y_uncond], dim=0)

        # 3. Model forward pass
        # Output shape: [2B, img_channel, H, W] (directly predicts velocity field)
        v_pred = self.forward(x_in, t_in, y_in)

        # 4. Split conditional and unconditional predictions
        # v_pred: [2B, C, H, W] -> cond [B, C, H, W], uncond [B, C, H, W]
        cond_v, uncond_v = v_pred.chunk(2, dim=0)

        # 5. Apply CFG formula: v = v_uncond + scale * (v_cond - v_uncond)
        half_v = uncond_v + cfg_scale * (cond_v - uncond_v)

        return half_v


if __name__ == "__main__":
    # Configuration
    BATCH_SIZE = 4
    TIMESTEPS = 100

    # Create model
    model = FlowMatchingModel(3, 28, 28, 4, 128)

    x = torch.randn(BATCH_SIZE, 3, 28, 28)
    t = torch.rand(BATCH_SIZE)  # Flow matching typically uses t in [0, 1]
    y = torch.randint(0, 10, (BATCH_SIZE,))

    output = model(x, t, y)

    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")
