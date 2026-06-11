import torch
import torch.nn as nn
import torch.nn.functional as F


class ECIRMOperator(nn.Module):
    """
    Mask-Driven Orthogonal Refining Operator.
    Rigorously disentangles causal invariant features from environmental noise
    using orthogonal constraints and Energy-Conditioned Invariant Risk Minimization.
    """

    def __init__(self, feature_dim: int, num_classes: int, lambda_irm: float = 1.0, beta_ortho: float = 0.5):
        super(ECIRMOperator, self).__init__()
        self.classifier = nn.Linear(feature_dim, num_classes)
        self.lambda_irm = lambda_irm
        self.beta_ortho = beta_ortho

    def orthogonal_penalty(self, z_c: torch.Tensor, z_e: torch.Tensor) -> torch.Tensor:
        """
        Enforces linear independence between causal (z_c) and spurious (z_e) features.
        Calculates the normalized cosine similarity / dot product penalty.
        """
        # Normalize representations to prevent scale domination
        z_c_norm = F.normalize(z_c, p=2, dim=-1)
        z_e_norm = F.normalize(z_e, p=2, dim=-1)

        # Mean squared dot product to enforce orthogonality (z_c_norm \perp z_e_norm)
        ortho_loss = torch.mean((z_c_norm * z_e_norm).sum(dim=-1) ** 2)
        return ortho_loss

    def irm_penalty(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Calculates the Invariant Risk Minimization (IRM) gradient penalty.
        Uses the dummy classifier 'w' scale trick to compute gradient norm.
        """
        dummy_w = torch.tensor(1.0, requires_grad=True, device=logits.device)
        scaled_logits = logits * dummy_w

        if logits.shape[-1] == 1:
            loss = F.mse_loss(scaled_logits.squeeze(), targets.float())  # Regression
        else:
            loss = F.cross_entropy(scaled_logits, targets.long())  # Classification

        # Compute gradient with respect to dummy w
        grad = torch.autograd.grad(loss, [dummy_w], create_graph=True)[0]

        # Penalty is the squared L2 norm of the gradient
        penalty = torch.sum(grad ** 2)
        return penalty

    def forward(self, features: torch.Tensor, mask: torch.Tensor, targets: torch.Tensor) -> Tuple[torch.Tensor, dict]:
        """
        Executes the heavy-duty data purification on structurally biased data.
        """
        # 1. Feature Disentanglement
        z_c = features * mask  # Causal invariant features
        z_e = features * (1.0 - mask)  # Spurious environmental noise

        # 2. Prediction based strictly on causal features
        logits = self.classifier(z_c)

        # 3. Objective Components
        # Empirical Risk Minimization (ERM)
        if logits.shape[-1] == 1:
            loss_erm = F.mse_loss(logits.squeeze(), targets.float())
        else:
            loss_erm = F.cross_entropy(logits, targets.long())

        # Orthogonal Constraint
        loss_ortho = self.orthogonal_penalty(z_c, z_e)

        # IRM Penalty
        loss_irm = self.irm_penalty(logits, targets)

        # 4. Total Meta-Causal Objective
        total_loss = loss_erm + self.lambda_irm * loss_irm + self.beta_ortho * loss_ortho

        loss_dict = {
            'loss_total': total_loss.item(),
            'loss_erm': loss_erm.item(),
            'loss_irm': loss_irm.item(),
            'loss_ortho': loss_ortho.item()
        }

        return total_loss, loss_dict