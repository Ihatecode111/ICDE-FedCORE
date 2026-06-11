import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Dict, Optional, Tuple


class RegimeAwarePrompting(nn.Module):
    """
    Regime-Aware Meta-Causal Prompting Module.
    Dynamically couples the global ultra-low-dimensional prompt with local
    environmental signatures to generate a client-specific feature mask.
    """

    def __init__(self, prompt_dim: int, feature_dim: int):
        super(RegimeAwarePrompting, self).__init__()
        self.prompt_dim = prompt_dim
        self.feature_dim = feature_dim

        # Projection matrices mapping low-dim prompt to feature space
        self.W_p = nn.Linear(prompt_dim, feature_dim, bias=False)
        self.W_s = nn.Linear(feature_dim, feature_dim, bias=False)

    def forward(self, global_prompt: torch.Tensor, local_signature: torch.Tensor) -> torch.Tensor:
        """
        Args:
            global_prompt (Tensor): p in R^m, the global causal prior.
            local_signature (Tensor): s, the local environmental context.
        Returns:
            mask (Tensor): m_c, the dynamic feature mask in [0, 1].
        """
        # Element-wise fusion and sigmoid activation to bound mask values
        fused_state = self.W_p(global_prompt) + self.W_s(local_signature)
        mask = torch.sigmoid(fused_state)
        return mask


class FedCOREServer:
    """
    Global Meta-Causal Orchestration Server.
    Strictly constrained to O(m) communication complexity. Bypasses full-parameter sync.
    """

    def __init__(self, prompt_dim: int, device: str = 'cpu'):
        self.prompt_dim = prompt_dim
        self.device = torch.device(device)
        # Initialize the global prompt repository
        self.global_prompt = torch.randn(prompt_dim, device=self.device, requires_grad=True)

    def aggregate_prompts(self, client_prompts: List[torch.Tensor], weights: List[float]) -> None:
        """
        Federated aggregation of only the lightweight meta-causal prompts.
        """
        assert len(client_prompts) == len(weights), "Mismatched prompts and weights."
        total_weight = sum(weights)
        normalized_weights = [w / total_weight for w in weights]

        aggregated_prompt = torch.zeros_like(self.global_prompt)
        for p, w in zip(client_prompts, normalized_weights):
            aggregated_prompt += w * p.to(self.device)

        self.global_prompt.data.copy_(aggregated_prompt)

    def broadcast(self) -> torch.Tensor:
        """Returns the O(m) payload to edge clients."""
        return self.global_prompt.detach().clone()