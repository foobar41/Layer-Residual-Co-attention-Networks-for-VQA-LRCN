import torch.nn as nn
import torch
from components.attention import SelfAttention, GuidedAttention

class PureStackingLRCN(nn.Module):
    """
    A layer residual network with a pure-stacking inspired architecture proposed in the LRCN paper.

    This model progressively feeds the output of question self-attention
    to the guided attention up to the L-th layer, while maintaining 
    layer-residual connections.
    """
    def __init__(
            self,
            num_heads: int=8,
            hidden_dim: int=64,
            num_layers: int=6,
    ):
        super(PureStackingLRCN, self).__init__()

        ## L layers of self-attention for images
        self.img_sa_layers = nn.ModuleList([
            SelfAttention(num_heads=num_heads, hidden_dim=hidden_dim) for _ in range(num_layers)
        ])

        ## L layers of self-attention for questions
        self.q_sa_layers = nn.ModuleList([
            SelfAttention(num_heads=num_heads, hidden_dim=hidden_dim) for _ in range(num_layers)
        ])

        ## L layers of guided attention for image question interaction
        self.img_ga_layers = nn.ModuleList([
            GuidedAttention(num_heads=num_heads, hidden_dim=hidden_dim) for _ in range(num_layers)
        ])

        self.num_layers = num_layers
    
    def forward(self,
            img_features: torch.Tensor,
            text_features: torch.Tensor,
            output_hidden_states: bool=False,
            output_attn_weights: bool=False)-> dict:
        """
        Args:
            img_features (torch.Tensor): Visual features X of shape (batch_size, num_regions, hidden_dim)
            text_features (torch.Tensor): Question features Y of shape (batch_size, seq_len, hidden_dim)
            output_hidden_states (bool): Store hidden states or not
            
        Returns:
            dict: (final_img_features, final_question_features, output_hidden_states(Optional)) after L layers of processing
        """

        ## Initializing layer-residual connections
        ## From paper, X^(0) = X, Y^(0) = Y, and PrevRe^(0) = X for first layer
        prev_img_sa_output = img_features

        current_img = img_features
        current_q = text_features

        ## Storing hidden states
        question_hidden_states = [current_q]
        image_hidden_states = [current_img]

        for layer_idx in range(self.num_layers):
            ## 1. SA for img with layer residual connection
            img_sa_output, img_attn_wts_out_sa = self.img_sa_layers[layer_idx](
                x = current_img, 
                prev_sa_output = prev_img_sa_output
            )

            ## 2. SA for question
            q_sa_output, q_attn_wts_out_sa = self.q_sa_layers[layer_idx](x = current_q)

            ## 3. GA for img guided by question
            img_ga_output, img_attn_wts_out_ga = self.img_ga_layers[layer_idx](
                q = img_sa_output,
                k = q_sa_output
            )

            ## Storing hidden states
            image_hidden_states.append(img_ga_output)
            question_hidden_states.append(q_sa_output)

            ## Update layer residual for next layer
            prev_img_sa_output = img_sa_output

            ## Update current q and img
            current_img = img_ga_output
            current_q = q_sa_output

        outputs = {
            'image_features': current_img,
            'text_features': current_q
        }
        if output_hidden_states:
            outputs['hidden_states'] = {
                'image_hidden_states': image_hidden_states,
                'text_hidden_states': question_hidden_states
            }
        
        if output_attn_weights:
            outputs['attn_weights'] = {
                'image_self': img_attn_wts_out_sa,
                'image_guided': img_attn_wts_out_ga,
                'text_self': q_attn_wts_out_sa,
            }
        return outputs


