import torch.nn as nn
import torch

class SelfAttention(nn.Module):
    """
    Self attention block used in the original Layerresidual Co-Attention Network (LRCN) paper.
    """

    def __init__(
            self,
            num_heads: int=8,
            hidden_dim: int=64,

    ):
        super(SelfAttention, self).__init__()
        self.mha = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            batch_first=True
        )
        self.mha_norm = nn.LayerNorm(hidden_dim)
        self.ff = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        self.ff_norm = nn.LayerNorm(hidden_dim)

    def forward(
            self,
            x: torch.Tensor,
            prev_sa_output: torch.Tensor = None
    ):
        """
        Args:
            x (torch.Tensor): The input tensor of shape (batch_size, token_sequence_length, hidden_dim)
            prev_sa_output (torch.Tensor): The output of the previous self attention block of shape (batch_size, token_sequence_length, hidden_dim)
        Returns:
            The output tensor of the self attention block of shape (batch_size, token_sequence_length, hidden_dim)
        """
        mha_output, attn_wts = self.mha(query=x, key=x, value=x, need_weights=True)
        if prev_sa_output is not None:
            added_mha_output = mha_output + prev_sa_output # Layer Residual Mechanism - Adding previous layer output
        else:
            added_mha_output = mha_output
        norm_mha_output = self.mha_norm(added_mha_output) # First Layer Normalization
        ff_output = self.ff(norm_mha_output)
        added_ff_output = ff_output + norm_mha_output # Intermiediate skip connection
        norm_ff_output = self.ff_norm(added_ff_output) # Second Layer Normalization
        return norm_ff_output, attn_wts


class GuidedAttention(nn.Module):
    """
    Guided attention block used in the original Layerresidual Co-Attention Network (LRCN) paper.
    """

    def __init__(
            self,
            num_heads: int=8,
            hidden_dim: int=64,

    ):
        super(GuidedAttention, self).__init__()
        self.mha = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            batch_first=True
        )
        self.mha_norm = nn.LayerNorm(hidden_dim)
        self.ff = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        self.ff_norm = nn.LayerNorm(hidden_dim)

    def forward(
            self,
            q: torch.Tensor,
            k: torch.Tensor,
            prev_ga_output: torch.Tensor = None
    ):
        """
        Args:
            q (torch.Tensor): The input query tensor of shape (batch_size, token_sequence_length, hidden_dim).
            k (torch.Tensor): The input key tensor of shape (batch_size, token_sequence_length, hidden_dim). This tensor is used to guide the encoding of the query tensor.
            prev_ga_output (torch.Tensor): The output of the previous guided attention block of shape (batch_size, token_sequence_length, hidden_dim)
        Returns:
            The output tensor of the guided attention block of shape (batch_size, token_sequence_length, hidden_dim)
        """
        mha_output, attn_wts = self.mha(query=q, key=k, value=k, need_weights=True)
        if prev_ga_output is not None:
            added_mha_output = mha_output + prev_ga_output # Layer Residual Mechanism - Adding previous layer output
        else:
            added_mha_output = mha_output
        norm_mha_output = self.mha_norm(added_mha_output) # First Layer Normalization
        ff_output = self.ff(norm_mha_output)
        added_ff_output = ff_output + norm_mha_output # Intermiediate skip connection
        norm_ff_output = self.ff_norm(added_ff_output) # Second Layer Normalization
        return norm_ff_output, attn_wts