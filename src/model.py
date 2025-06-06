import torch
import torch.nn as nn
from enum import Enum
from typing import Union, Dict, Optional
from components.lrcn_costacking import CoStackingLRCN
from components.lrcn_enc_dec import EncoderDecoderLRCN
from components.lrcn_pure_stacking import PureStackingLRCN
from components.multimodal_fusion import MultimodalFusion
import numpy as np
from components.constants import EMBEDDING_MATRIX

class LRCN(nn.Module):
    """
    Layer-Residual Co-Attention Network (LRCN) with flexible architecture selection.
    
    This model combines one of three architecture variants (Pure-Stacking, Co-Stacking,
    or Encoder-Decoder) with multimodal fusion for VQA tasks.
    """
    def __init__(
            self,
            architecture_type: str = "pure_stacking",
            hidden_dim: int = 512,
            num_heads: int = 8,
            num_layers: int = 6,
            num_answers: int = 3129,
            dropout_rate: float = 0.1,
            embedding_matrix_path: str = f"./data/VQAv2/text_features/{EMBEDDING_MATRIX}"
    ):

        super(LRCN, self).__init__()

        ## Remaining preprocessing
        ## Create Glove Embeddings (Paper says they should be trainable embeddings)
        # Create embedding matrix from GloVe
        embedding_matrix = np.load(embedding_matrix_path, allow_pickle=True)

        # Add embedding as first layer in model
        self.embed = nn.Embedding.from_pretrained(
            torch.from_numpy(embedding_matrix), freeze=True
        )

        ## Text LSTM preprocessing
        self.lstm = torch.nn.LSTM(300, 512, batch_first=True)   ## For text

        ## Image convolution and linear preprocessing
        self.downsample = nn.Conv2d(2048, 2048, kernel_size=2, stride=2)    ## For images
        self.projection = nn.Linear(2048, 512)
        
        if architecture_type == "pure_stacking":
            self.feature_extractor = PureStackingLRCN(
                num_heads=num_heads,
                hidden_dim=hidden_dim,
                num_layers=num_layers
            )
        elif architecture_type == "co_stacking":
            self.feature_extractor = CoStackingLRCN(
                num_heads=num_heads,
                hidden_dim=hidden_dim,
                num_layers=num_layers
            )
        elif architecture_type == "encoder_decoder":
            self.feature_extractor = EncoderDecoderLRCN(
                num_heads=num_heads,
                hidden_dim=hidden_dim,
                num_layers=num_layers
            )
        else:
            raise ValueError(f"Unsupported architecture type: {architecture_type}")

        # Initialize multimodal fusion component
        self.fusion = MultimodalFusion(
            feature_dim=hidden_dim,
            output_dim=hidden_dim,
            vocab_size=num_answers,
            dropout_rate=dropout_rate
        )
        
        self.architecture_type = architecture_type
        self.hidden_dim = hidden_dim

    def forward(
            self,
            img_features: torch.Tensor,
            text_features: torch.Tensor,
            output_hidden_states: bool = False,
            output_attn_weights: bool=False
        ):
        """
        Forward pass through the complete LRCN model.
        
        Args:
            img_features: Image features of shape [batch_size, num_regions, hidden_dim]
            text_features: Question features of shape [batch_size, seq_length, 300]
            targets: Optional answer targets for loss calculation
            output_hidden_states: Whether to return intermediate hidden states
        
        Returns:
            Dictionary containing:
                'scores': Answer probability scores [batch_size, num_answers]
                'hidden_states': Intermediate states (if requested)
        """
        ## Remaining preprocessing IMAGES
        ## Downsample and project
        batch_size = img_features.shape[0]
        img_features = self.downsample(img_features)

        # Permute to [1, 8, 8, 2048] for linear layer
        img_features = img_features.permute(0, 2, 3, 1).contiguous()

        # Flatten the grid features before FC layer
        img_features = img_features.view(batch_size, -1, 2048) # TODO: Explore if any better way instead of hard coding
        # Apply linear projection [1, 8, 8, 512]
        img_features = self.projection(img_features)

        ## Remaining preprocessing TEXT
        text_features = self.embed(text_features)  # [batch, 14, 300]
        text_features, _ = self.lstm(text_features) # Converts to [batch_size, seq_length, 512]

        ## Select architecture
        feature_outputs = self.feature_extractor(img_features = img_features, text_features=text_features, output_hidden_states=output_hidden_states, output_attn_weights=output_attn_weights)

        ## Extract processed features
        visual_features = feature_outputs['image_features']
        textual_features = feature_outputs['text_features']

        ## Apply fusion and classification
        scores = self.fusion(visual_features, textual_features)

        outputs = {'scores': scores}

        # Include hidden states and attention weights if requested
        if output_hidden_states and 'hidden_states' in feature_outputs:
            outputs['hidden_states'] = feature_outputs['hidden_states']
        if output_attn_weights and 'attn_weights' in feature_outputs:
            outputs['attn_weights'] = feature_outputs['attn_weights']

        return outputs
