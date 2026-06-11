import torch
import torch.nn as nn


class CWAZSniffer(nn.Module):
    """
    Meta-Driven CWAZ Sniffer.
    Acts as an O(1) dynamic router to sustain high edge throughput by bypassing
    clean/routine instances and strictly isolating long-tail structural biases.
    """

    def __init__(self, initial_tau: float = 0.5, ema_momentum: float = 0.9, epsilon: float = 1e-6):
        super(CWAZSniffer, self).__init__()
        # Use buffers so threshold isn't treated as a trainable parameter
        self.register_buffer('tau', torch.tensor(initial_tau))
        self.momentum = ema_momentum
        self.epsilon = epsilon

    def compute_spurious_energy(self, features: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """
        Calculates the spurious energy Sc(x) by evaluating the overlapping
        footprint of features against the inverted causal mask.
        """
        # H_spurious = h * (1 - m^c)
        spurious_components = features * (1.0 - mask)

        # Calculate L1 norm energy ratio
        l1_spurious = torch.norm(spurious_components, p=1, dim=-1)
        l1_total = torch.norm(features, p=1, dim=-1) + self.epsilon

        energy = l1_spurious / l1_total
        return energy

    def update_threshold(self, batch_energy: torch.Tensor) -> None:
        """
        Dynamically updates the adaptive routing threshold via EMA.
        """
        mean_energy = batch_energy.mean().detach()
        new_tau = self.momentum * self.tau + (1.0 - self.momentum) * mean_energy
        self.tau.copy_(new_tau)

    def forward(self, features: torch.Tensor, mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Evaluates input instances and dynamically routes them.

        Returns:
            Tuple containing indices of 'clean' (bypass) and 'dirty' (route to operator) data.
        """
        energy = self.compute_spurious_energy(features, mask)

        # Dynamic routing condition
        dirty_indices = torch.nonzero(energy >= self.tau, as_tuple=True)[0]
        clean_indices = torch.nonzero(energy < self.tau, as_tuple=True)[0]

        # Adaptive threshold update during training
        if self.training:
            self.update_threshold(energy)

        return clean_indices, dirty_indices