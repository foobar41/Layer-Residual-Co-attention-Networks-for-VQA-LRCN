import torch.nn as nn
import torch
from components.attention import SelfAttention, GuidedAttention

class EncoderDecoderLRCN(nn.Module):
    """
    A layer residual network with a encoder-decoder inspired architecture proposed in the LRCN paper.
    """
    def __init__(
            self,
            num_heads: int=8,
            hidden_dim: int=64,
            num_layers: int=6,
    ):
        """
        
        """
        super(EncoderDecoderLRCN, self).__init__()
        self.text_encoder = nn.ModuleList([
            SelfAttention(num_heads=num_heads, hidden_dim=hidden_dim) for _ in range(num_layers)
        ])
        self.image_decoder_sa = nn.ModuleList([
            SelfAttention(num_heads=num_heads, hidden_dim=hidden_dim) for _ in range(num_layers)
        ])
        self.image_decoder_ga = nn.ModuleList([
            GuidedAttention(num_heads=num_heads, hidden_dim=hidden_dim) for _ in range(num_layers)
        ])
    
    def forward(
            self,
            img_features: torch.Tensor,
            text_features: torch.Tensor,
            output_hidden_states: bool=False
    ):
        """

        """
        # Encode the text features completely (Encoder)
        text_features_output = text_features
        text_hidden_states = [text_features_output]
        for sa_layer in self.text_encoder:
            text_features_output = sa_layer(text_features_output)
            text_hidden_states.append(text_features_output)
        # Decode image features with encoded text features as guidance (Decoder)
        PrevRe = img_features
        img_features_output = img_features
        image_hidden_states = [img_features_output]
        for sa_layer, ga_layer in zip(self.image_decoder_sa, self.image_decoder_ga):
            sa_output = sa_layer(
                x=img_features_output,
                prev_sa_output=PrevRe
            )
            ga_output = ga_layer(
                q=sa_output,
                k=text_features_output
            )
            img_features_output = ga_output
            image_hidden_states.append(img_features_output)
            PrevRe = sa_output
        outputs = {
            'image_features': img_features_output,
            'text_features': text_features_output
        }
        if output_hidden_states:
            outputs['hidden_states'] = {
                'image_hidden_states': image_hidden_states,
                'text_hidden_states': text_hidden_states
            }
        return outputs