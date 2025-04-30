import torch
import torch.nn as nn
import torch.nn.functional as F

class AttentionPooling(nn.Module):
    """
    Implements attention pooling mechanism as described in equation 15-16.
    Produces attention weights using MLP and applies weighted pooling to features.
    """

    def __init__(self, feature_dim=512, dropout_rate=0.1):
        super(AttentionPooling, self).__init__()

        # MLP implementation: FC(D)-ReLU-Dropout(0.1)-FC(1)
        self.mlp = nn.Sequential(
            nn.Linear(feature_dim, feature_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(feature_dim, 1)
        )
    
    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """
        Args:
            features: Tensor of shape [batch_size, num_regions/words, feature_dim] (could be img or text)
        Returns:
            pooled_features: Tensor of shape [batch_size, feature_dim]
        """

        ## Calculate attention weights alpha (images) and beta (text)
        attn_weights = self.mlp(features)   # Shape: [batch_size, num_regions/words, 1]

        ## applying softmax
        attn_weights = F.softmax(attn_weights, dim=1)

        ## applying attention weights to features
        pooled_features = torch.sum(features * attn_weights, dim=1)

        return pooled_features
    
class ModalityFusion(nn.Module):
    """
    Implements feature fusion as described in equation 17.
    Combines visual and textual features using linear projections and layer normalization.
    """

    def __init__(self, input_dim=512, output_dim=512):
        super(ModalityFusion, self).__init__()

        self.visual_proj = nn.Linear(input_dim, output_dim)
        self.textual_proj = nn.Linear(input_dim, output_dim)

        self.layer_norm = nn.LayerNorm(output_dim)
        self.output_proj = nn.Linear(output_dim, output_dim)

    def forward(self, visual_features, textual_features):
        """
        Args:
            visual_features: Tensor of shape [batch_size, feature_dim]
            textual_features: Tensor of shape [batch_size, feature_dim]
        Returns:
            fused_features: Tensor of shape [batch_size, output_dim]
        """
        ## Projecting each modality
        visual_proj = self.visual_proj(visual_features)
        textual_proj = self.textual_proj(textual_features)

        ## Combining and normalizing
        fused = self.layer_norm(visual_proj + textual_proj)

        ## f = zW_z (final projection)
        output = self.output_proj(fused)

        return output
    
class AnswerClassifier(nn.Module):
    """
    Implements answer classification as described in equations 18-19.
    Projects fused features to answer space and applies activation functions.
    """
    def __init__(self, input_dim=512, vocab_size=3129):
        super(AnswerClassifier, self).__init__()

        self.classifier = nn.Linear(input_dim, vocab_size)

    def forward(self, fused_features):
        """
        Args:
            fused_features: Tensor of shape [batch_size, feature_dim]
        Returns:
            logits: Tensor of shape [batch_size, vocab_size]
            scores: Tensor of shape [batch_size, vocab_size] (after sigmoid)
        """

        ## Apply ReLU activation
        hidden = F.relu(fused_features)

        ## Project to answer vocab size
        logits = self.classifier(hidden)

        ## Apply sigmoid to get probabilities
        scores = torch.sigmoid(logits)

        return logits, scores