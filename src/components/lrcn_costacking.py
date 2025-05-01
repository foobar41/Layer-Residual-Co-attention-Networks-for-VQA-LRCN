import torch.nn as nn
import torch
from attention import SelfAttention, GuidedAttention

class CoStackingLRCN(nn.Module):
    """
    A layer residual network with a pure-stacking inspired architecture proposed in the LRCN paper.

    This architecture implements bidirectional guided attention where:
    1. Image features guide question features
    2. Question features guide image features
    
    Each modality maintains its own layer-residual connections.
    """

    def __init__(
            self,
            num_heads: int=8,
            hidden_dim: int=64,
            num_layers: int=6,
    ):
        ## Self Attention blocks for image features
        self.img_sa_layers = nn.ModuleList([
            SelfAttention(num_heads=num_heads, hidden_dim=hidden_dim) for _ in range(num_layers)
        ])

        ## Self Attention blocks for question features
        self.q_sa_layers = nn.ModuleList([
            SelfAttention(num_heads=num_heads, hidden_dim=hidden_dim)
            for _ in range(num_layers)
        ])

        ## Guided Attention blocks for image features
        self.img_ga_layers = nn.ModuleList([
            GuidedAttention(num_heads=num_heads, hidden_dim=hidden_dim)
            for _ in range(num_layers)
        ])

        ## Guided Attention blocks for question features
        self.q_ga_layers = nn.ModuleList([
            GuidedAttention(num_heads=num_heads, hidden_dim=hidden_dim)
            for _ in range(num_layers)
        ])

        self.num_layers = num_layers
    
    def forward(self, img_features: torch.Tensor,
            question_features: torch.Tensor,
            output_hidden_states: bool=False)-> dict:
        """
        Args:
            img_features (torch.Tensor): Visual features X of shape (batch_size, num_regions, hidden_dim)
            question_features (torch.Tensor): Question features Y of shape (batch_size, seq_len, hidden_dim)
            output_hidden_states (bool): Store hidden states or not
            
        Returns:
            dict: (final_img_features, final_question_features, output_hidden_states(Optional)) after L layers of processing
        """

        ## Initializing layer residual connections
        # Following paper notation: X^(0) = X, Y^(0) = Y
        # For first layer: PrevReX^(0) = X, PrevReY^(0) = Y
        prev_img_sa_output = img_features
        prev_q_sa_output = question_features

        current_img = img_features
        current_q = question_features

        ## Storing hidden states
        question_hidden_states = [current_q]
        image_hidden_states = [current_img]

        ## Process through L layers
        for layer_idx in range(self.num_layers):
            ## 1. Self Attention for IMAGE with layer residuals
            img_sa_output = self.img_sa_layers[layer_idx](
                x = current_img,
                prev_sa_output = prev_img_sa_output
            )

            ##2. Self Attention for QUESTION with layer residuals
            q_sa_output = self.q_sa_layers[layer_idx](
                x = current_q,
                prev_sa_output = prev_q_sa_output
            )

            ## 3. Guided Attention for QUESTION (guided by img SA output)
            q_ga_output = self.q_ga_layers[layer_idx](
                q = q_sa_output,
                k = img_sa_output
            )

            ## 4. Guided attention for IMAGE (guided by question)
            img_ga_output = self.img_ga_layers[layer_idx](
                q = img_sa_output,
                k = q_ga_output
            )

            ## Updating layer residual connections
            prev_img_sa_output = img_sa_output
            prev_q_sa_output = q_sa_output

            ## Storing hidden states
            image_hidden_states.append(img_ga_output)
            question_hidden_states.append(q_ga_output)

            ## Updating current features for next layer
            current_img = img_ga_output
            current_q = q_ga_output
        
        outputs = {
            'image_features': current_img,
            'text_features': current_q
        }
        if output_hidden_states:
            outputs['hidden_states'] = {
                'image_hidden_states': image_hidden_states,
                'text_hidden_states': question_hidden_states
            }
        return outputs

