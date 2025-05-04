import torch.nn as nn
from components.multimodal_fusion_components import ModalityFusion, AttentionPooling, AnswerClassifier

class MultimodalFusion(nn.Module):
    """
    Main class that implements the complete multimodal fusion pipeline.
    Combines attention pooling, feature fusion, and answer classification.
    """
    def __init__(self, feature_dim=512, output_dim=512, vocab_size=3129, dropout_rate=0.1):
        super(MultimodalFusion, self).__init__()

        ## Attention Pooling for each modality
        self.visual_attention = AttentionPooling(feature_dim, dropout_rate)
        self.textual_attention = AttentionPooling(feature_dim, dropout_rate)

        ## Feature fusion
        self.fusion = ModalityFusion(feature_dim, output_dim)

        ## Answer Classifier
        self.classifier = AnswerClassifier(output_dim, vocab_size)
    
    def forward(self, visual_features, textual_features, targets=None):
        """
        Args:
            visual_features: Tensor of shape [batch_size, num_regions, feature_dim]
            textual_features: Tensor of shape [batch_size, seq_length, feature_dim]
            targets: Optional target tensor of shape [batch_size, vocab_size]
        
        Returns:
            scores: Answer probability scores
            loss: BCE loss if targets provided, else None
        """
        pooled_visual = self.visual_attention(visual_features)
        pooled_textual = self.textual_attention(textual_features)

        fused_features = self.fusion(pooled_visual, pooled_textual)

        _, scores = self.classifier(fused_features)

        return scores
