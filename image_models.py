"""Provide modules for building vision transformers.

This module provides core building blocks for transformer-based
vision models, including patch embedding/unembedding, positional
encoding, multi-head attention, and Mlp layers. It is designed to
be compatible with both ViT and DiT architectures.
"""

from typing import Optional
import torch
import torch.nn as nn


__all__ = [
    "PatchEmbedder",
    "PatchUnembedder",
    "PositionalEmbedder",
    "Attention",
    "Mlp",
    "AdaLNZeroBlock",
    "modulate",
]


class PatchEmbedder(nn.Module):
    """Converts an image into a sequence of flattened patches.

    This layer performs patchification by applying a convolution
    operation with kernel size and stride equal to the patch size,
    effectively dividing the input image into non-overlapping patches
    and projecting them into a specified embedding dimension.
    """

    def __init__(self, in_channels: int, embed_dim: int,
                 patch_size: int):
        """Initialize the PatchEmbedder layer.

        Args:
            in_channels: Number of input channels (e.g., 3 for RGB).
            embed_dim: Dimension of the output patch embeddings.
            patch_size: Size of each patch (assumed square).
        """
        super().__init__()
        self.patch_size = patch_size
        # Conv2d layer that projects patches into the embedding space
        self.proj = nn.Conv2d(
            in_channels,
            embed_dim,
            kernel_size=patch_size,
            stride=patch_size
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Convert input image into patch embeddings.

        Pads the input if necessary to ensure complete patches, then
        applies the convolution projection and reshapes the output.

        Args:
            x: Input tensor of shape [batch_size, in_channels,
                img_height, img_width].

        Returns:
            Tensor of shape [batch_size, seq_len, embed_dim].
        """
        batch_size, in_channels, img_height, img_width = x.shape

        # Manually pad to achieve ceiling division for patch count
        pad_h = ((self.patch_size - img_height % self.patch_size) %
                 self.patch_size)
        pad_w = ((self.patch_size - img_width % self.patch_size) %
                 self.patch_size)
        # Pad right and bottom
        x = nn.functional.pad(x, (0, pad_w, 0, pad_h))
        x = self.proj(x)
        x = x.flatten(2).transpose(1, 2)
        return x


class PatchUnembedder(nn.Module):
    """Convert patch embeddings back to an image.

    This layer performs the inverse operation of PatchEmbedder. It
    takes a sequence of patch embeddings and reconstructs the original
    image by reshaping and applying a linear projection to each patch.
    """

    def __init__(
        self,
        embed_dim: int,
        patch_size: int,
        out_channels: int,
        img_height: int,
        img_width: int
    ):
        """Initialize the PatchUnembedder layer.

        Args:
            embed_dim: Dimension of input patch embeddings.
            patch_size: Size of each patch (assumed square).
            out_channels: Number of output channels.
            img_height: Original image height.
            img_width: Original image width.
        """
        super().__init__()
        self.patch_size = patch_size
        self.out_channels = out_channels
        self.img_height = img_height
        self.img_width = img_width

        # Compute patch grid dimensions at initialization time
        pad_h = (patch_size - img_height % patch_size) % patch_size
        pad_w = (patch_size - img_width % patch_size) % patch_size
        self.padded_height = img_height + pad_h
        self.padded_width = img_width + pad_w
        self.num_patches_h = self.padded_height // patch_size
        self.num_patches_w = self.padded_width // patch_size

        # Linear layer to project embeddings back to pixel values
        self.linear = nn.Linear(
            embed_dim,
            patch_size * patch_size * out_channels
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Reconstruct an image from patch embeddings.

        Args:
            x: Tensor of shape [batch_size, seq_len, embed_dim].

        Returns:
            Tensor of shape [batch_size, out_channels, img_height,
            img_width], the reconstructed image (cropped to original
            dimensions if padding was applied).
        """
        batch_size, seq_len, embed_dim = x.shape

        # Step 1: Linear projection from embedding to pixel dimension
        x = self.linear(x)

        # Step 2: Reshape and permute dimensions to reconstruct image
        x = x.reshape(
            batch_size,
            self.num_patches_h,
            self.num_patches_w,
            self.patch_size,
            self.patch_size,
            self.out_channels
        )
        # Adjust dimension order
        x = x.permute(0, 5, 1, 3, 2, 4)
        x = x.reshape(
            batch_size,
            self.out_channels,
            self.padded_height,
            self.padded_width
        )
        x = x[:, :, :self.img_height, :self.img_width]
        return x


class PositionalEmbedder(nn.Module):
    """Sinusoidal positional encoding for transformer models.

    Implements non-learnable sinusoidal positional embeddings where
    even dimensions use sine and odd dimensions use cosine, following
    the original "Attention Is All You Need" paper.
    """

    def __init__(self, embed_dim: int, max_len: int = 10000):
        """Initialize the positional embedder.

        Args:
            embed_dim: Dimension of the embeddings.
            max_len: Maximum sequence length to pre-compute encodings for.
        """
        super().__init__()
        self.embed_dim = embed_dim
        self.max_len = max_len

        # Pre-compute positional encoding matrix
        pos_encoding_table = torch.zeros(max_len, embed_dim)
        # Shape of `position`: [max_len, 1]
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)

        half_dim = embed_dim // 2

        freqs = torch.exp(
            torch.arange(0, half_dim).float() *
            (-torch.log(torch.tensor(10000.0)) / embed_dim)
        )

        pos_encoding_table[:, :half_dim] = torch.sin(position * freqs)
        pos_encoding_table[:, half_dim:] = torch.cos(position * freqs)

        # Add batch dimension: [1, max_len, embed_dim]
        # Use register_buffer so it moves with the model .to(device)
        self.register_buffer('pos_encoding_table', pos_encoding_table.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Add positional encoding to input embeddings.

        Args:
            x: Input tensor of shape [batch_size, seq_len, embed_dim].

        Returns:
            Tensor of shape [batch_size, seq_len, embed_dim] with
            positional information added.
        """
        batch_size, seq_len, embed_dim = x.shape

        # Slice the pre-computed positional encoding to match seq_len
        # No need for .to(x.device) because register_buffer handles device
        x = x + self.pos_encoding_table[:, :seq_len, :]
        return x


class Attention(nn.Module):
    """Multi-head attention module using PyTorch's built-in implementation.

    This is a minimal implementation of the attention mechanism commonly
    used in vision transformers (ViT) and diffusion models (DiT). It uses
    a fused QKV projection followed by PyTorch's native
    MultiHeadAttention.
    """

    def __init__(
        self,
        embed_dim: int,
        num_heads: int = 8,
        qkv_bias: bool = False,
        attn_drop: float = 0.,
    ) -> None:
        """Initialize the attention module.

        Args:
            embed_dim: Input feature dimension.
            num_heads: Number of attention heads.
            qkv_bias: Whether to include bias in QKV projection.
            attn_drop: Dropout rate for attention weights.
        """
        if embed_dim % num_heads:
            raise ValueError(
                f'embed_dim (get {embed_dim}) must be divisible'
                f'by num_heads (get {num_heads})')
        
        super().__init__()
        self.num_heads = num_heads
        # Dimension per head
        self.head_dim = embed_dim // num_heads
        # Total attention dimension (num_heads * head_dim)
        self.attn_dim = num_heads * self.head_dim

        # 1. Fused QKV linear layer (consistent with timm design)
        self.qkv_proj = nn.Linear(
            embed_dim,
            self.attn_dim * 3,
            bias=qkv_bias
        )

        # 2. PyTorch native MultiHeadAttention
        # Note: embed_dim corresponds to total attention dimension
        self.multi_head_attn = nn.MultiheadAttention(
            embed_dim=self.attn_dim,
            num_heads=num_heads,
            dropout=attn_drop,
            bias=False,  # QKV bias is handled in self.qkv_proj
            # Input format: [batch_size, seq_len, embed_dim]
            batch_first=True,
        )


    def forward(
        self,
        x: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Compute multi-head attention.

        Args:
            x: Input tensor of shape [batch_size, seq_len, embed_dim].
            attn_mask: Optional attention mask for the attention
                computation.

        Returns:
            Output tensor of shape [batch_size, seq_len, embed_dim].
        """
        batch_size, seq_len, embed_dim = x.shape

        # Step 1: QKV projection (fused, consistent with timm)
        qkv = self.qkv_proj(x).reshape(
            batch_size, seq_len, 3, self.attn_dim
        ).permute(2, 0, 1, 3)
        # q/k/v: [batch_size, seq_len, attn_dim]
        query, key, value = qkv.unbind(0)

        # Step 2: Compute attention using PyTorch MultiHeadAttention
        # Note: MultiHeadAttention handles head splitting internally
        attn_out, _ = self.multi_head_attn(
            query=query,
            key=key,
            value=value,
            attn_mask=attn_mask,
            # Skip returning attention weights for efficiency
            need_weights=False,
        )

        return attn_out


class Mlp(nn.Module):
    """Multi-layer perceptron (Mlp).

    This module implements a two-layer Mlp. It consists of a
    linear projection that expands the dimension, followed by
    activation and dropout, then a linear projection and dropout.
    """

    def __init__(
        self,
        in_features: int,
        hidden_features: Optional[int] = None,
        out_features: Optional[int] = None,
        dropout: float = 0.,
        activation: Optional[nn.Module] = None,
    ) -> None:
        """Initialize the Mlp module.

        Args:
            in_features: Input feature dimension.
            hidden_features: Hidden dimension (defaults to in_features).
            out_features: Output dimension (defaults to in_features).
            dropout: Dropout rate after each linear layer.
            activation: Activation function (defaults to GELU).
        """
        super().__init__()
        self.in_features = in_features
        self.hidden_features = hidden_features or in_features
        self.out_features = out_features or in_features

        self.fc1 = nn.Linear(self.in_features, self.hidden_features)
        self.activation = activation or nn.GELU()
        self.drop1 = nn.Dropout(dropout)

        self.fc2 = nn.Linear(self.hidden_features, self.out_features)
        self.drop2 = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Compute the Mlp forward pass.

        Args:
            x: Input tensor of shape [batch_size, seq_len, in_features].

        Returns:
            Tensor of shape [batch_size, seq_len, out_features].
        """
        x = self.fc1(x)
        x = self.activation(x)
        x = self.drop1(x)
        x = self.fc2(x)
        x = self.drop2(x)

        return x


def modulate(x: torch.Tensor, shift: torch.Tensor,
             scale: torch.Tensor) -> torch.Tensor:
    """Apply adaptive shift and scale to input tensor.

    This function implements the adaptive layer norm zero (adaLN-Zero)
    modulation mechanism used in DiT. It applies element-wise
    shift and scale to the input tensor.

    Args:
        x: Input tensor of shape [batch_size, seq_len, embed_dim].
        shift: Shift tensor of shape [batch_size, embed_dim].
        scale: Scale tensor of shape [batch_size, embed_dim].

    Returns:
        Modulated tensor of shape [batch_size, seq_len, embed_dim].
    """
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


class AdaLNZeroBlock(nn.Module):
    """Transformer block with adaLN-Zero conditioning.

    This block implements a transformer block with adaptive layer
    normalization and zero-initialized residual modulation, as
    described in the Scalable Diffusion Models with Transformers
    (DiT) paper. It consists of self-attention and Mlp layers,
    both modulated by a conditioning vector.
    """

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
    ) -> None:
        """Initialize the AdaLN-Zero block.

        Args:
            hidden_size: Hidden feature dimension.
            num_heads: Number of attention heads.
            mlp_ratio: Ratio of Mlp hidden dimension to hidden_size.
        """
        super().__init__()

        # Adaptive layer norm modulation
        # Projects conditioning vector to 6 * hidden_size
        # (shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 6 * hidden_size, bias=True)
        )

        # Layer norm before attention (without learnable parameters)
        self.norm1 = nn.LayerNorm(
            hidden_size,
            elementwise_affine=False,
            eps=1e-6
        )
        # Self-attention layer
        self.attn = Attention(
            embed_dim=hidden_size,
            num_heads=num_heads,
            qkv_bias=True,
        )
        # Layer norm before Mlp (without learnable parameters)
        self.norm2 = nn.LayerNorm(
            hidden_size,
            elementwise_affine=False,
            eps=1e-6
        )
        # Mlp with expanded hidden dimension
        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        self.mlp = Mlp(
            in_features=hidden_size,
            hidden_features=mlp_hidden_dim,
            # Using GELU same as DiT implementation.
            activation=nn.GELU(approximate="tanh")
        )
        self._initialize_weights()

    def _initialize_weights(self) -> None:
        """Initialize weights for zero-conditional modulation.

        Zero-initialize the last linear layer in adaLN_modulation,
        following the DiT paper to enable stable training with
        zero-initialized modulation.
        """
        # Zero-out adaLN modulation layers
        nn.init.constant_(self.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.adaLN_modulation[-1].bias, 0)

    def forward(
        self,
        x: torch.Tensor,
        condition: torch.Tensor
    ) -> torch.Tensor:
        """Compute the AdaLN-Zero block forward pass.

        Args:
            x: Input tensor of shape [batch_size, seq_len,
                hidden_features].
            condition: Conditioning tensor of shape [batch_size,
                hidden_features].

        Returns:
            Output tensor of shape [batch_size, seq_len,
            hidden_features].
        """
        # Compute modulation parameters from conditioning vector
        # Each parameter has shape [batch_size, hidden_features]
        (
            shift_msa, scale_msa, gate_msa,
            shift_mlp, scale_mlp, gate_mlp
        ) = self.adaLN_modulation(condition).chunk(6, dim=1)

        # Step 1: Adaptive attention with residual modulation
        x = x + gate_msa.unsqueeze(1) * self.attn(
            modulate(self.norm1(x), shift_msa, scale_msa)
        )

        # Step 2: Adaptive Mlp with residual modulation
        x = x + gate_mlp.unsqueeze(1) * self.mlp(
            modulate(self.norm2(x), shift_mlp, scale_mlp)
        )

        return x


# -------------------------- Test Code --------------------------
if __name__ == "__main__":
    # Test all components with sample input
    patch_embed = PatchEmbedder(3, 512, 16)
    pos_encoding = PositionalEmbedder(512)
    attention = Attention(512, 8)
    mlp = Mlp(in_features=512)
    patch_unembed = PatchUnembedder(512, 16, 3, 512, 220)
    ada_ln_zero_block = AdaLNZeroBlock(hidden_size=512, num_heads=8)

    # Initialize block weights (zero-initialize adaLN modulation)
    ada_ln_zero_block._initialize_weights()

    # Construct input and run forward pass
    x = torch.randn(2, 3, 512, 220)
    out_embed = patch_embed(x)
    out_pos = pos_encoding(out_embed)
    out_attn = attention(out_pos)
    out_mlp = mlp(out_attn)
    out_unembed = patch_unembed(out_mlp)

    # Test block with conditioning
    condition = torch.randn(2, 512)
    out_block = ada_ln_zero_block(out_mlp, condition)

    # Verify dimension consistency
    print(f"PatchEmbedder: {out_embed.shape}")
    print(f"PositionalEmbedder: {out_pos.shape}")
    print(f"Attention: {out_attn.shape}")
    print(f"Mlp: {out_mlp.shape}")
    print(f"PatchUnembedder: {out_unembed.shape}")
    print(f"AdaLNZeroBlock: {out_block.shape}")

    assert out_embed.shape == out_pos.shape == out_attn.shape
    assert out_attn.shape == out_mlp.shape
    assert out_mlp.shape == out_block.shape
    assert out_unembed.shape == x.shape
    print("All dimension validations passed")
