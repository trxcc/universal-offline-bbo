from typing import Sequence, Callable, Optional, Tuple, Union, Any

import torch
import torch.nn as nn


class SimpleMLP(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: Union[Sequence[int], int],
        output_dim: int = 1,
        require_batch_norm: bool = False,
    ) -> None:
        super(SimpleMLP, self).__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim

        self.require_batch_norm = require_batch_norm
        if require_batch_norm:
            self.batch_norm = nn.BatchNorm1d(input_dim)

        if isinstance(hidden_dim, int):
            hidden_dim = [hidden_dim]

        layers = []
        layers.append(nn.Linear(input_dim, hidden_dim[0]))
        layers.append(nn.ReLU())

        for i in range(len(hidden_dim) - 1):
            layers.append(nn.Linear(hidden_dim[i], hidden_dim[i + 1]))
            layers.append(nn.ReLU())

        layers.append(nn.Linear(hidden_dim[-1], output_dim))
        self.layers = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.require_batch_norm:
            x = self.batch_norm(x)
        return self.layers(x)
    
class DualHeadMLP(nn.Module):
    def __init__(
        self,
        hidden_dim: Sequence[int],
        input_dim: int,
        output_dim: int = 1,
        if_embedding: bool = False,
        activation: Callable[[torch.Tensor], torch.Tensor] = nn.ReLU(),
        initial_max_std: float = 0.2,
        initial_min_std: float = 0.1,
        final_activation: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
        std_activation: Optional[Callable[[torch.Tensor], torch.Tensor]] = None
    ):
        super().__init__()
        self.if_embedding = if_embedding
        self.activation = activation
        self.final_activation = final_activation
        self.std_activation = std_activation
        
        
        # Build shared layers
        layers = []
        for hidden_size in hidden_dim:
            layers.append(nn.Linear(input_dim, hidden_size))
            layers.append(activation)
            input_dim = hidden_size
        self.shared_layers = nn.Sequential(*layers)
        
        # Output heads
        self.mean_layer = nn.Linear(hidden_dim[-1], output_dim)
        self.logstd_layer = nn.Linear(hidden_dim[-1], output_dim)
        
        # Learnable max/min logstd parameters
        self.max_logstd = nn.Parameter(torch.full((1, 1), torch.log(torch.tensor(initial_max_std))))
        self.min_logstd = nn.Parameter(torch.full((1, 1), torch.log(torch.tensor(initial_min_std))))
        
        # Initialize weights using lecun uniform
        self._init_weights()

    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                # LeCun uniform initialization
                nn.init.kaiming_uniform_(module.weight, mode='fan_in', nonlinearity='linear')
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        
        x = x.reshape(x.shape[0], -1)
        
        # Shared layers
        hidden = self.shared_layers(x)
        
        # Mean head
        mean = self.mean_layer(hidden)
        if self.final_activation is not None:
            mean = self.final_activation(mean)
            
        # Logstd head
        logstd = self.logstd_layer(hidden)
        
        # Clamp logstd using softplus
        logstd = self.max_logstd - torch.nn.functional.softplus(self.max_logstd - logstd)
        logstd = self.min_logstd + torch.nn.functional.softplus(logstd - self.min_logstd)
        
        if self.std_activation is not None:
            logstd = self.std_activation(logstd)
        
        return mean, torch.exp(logstd)

    def get_distribution(self, obs: torch.Tensor) -> Any:
        """
        Returns a Normal distribution using the forward pass results.
        """
        from torch.distributions import Normal
        mean, std = self.forward(obs)
        return Normal(mean, std)

