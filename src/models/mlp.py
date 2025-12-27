from dataclasses import dataclass
import torch
import torch.nn as nn

class MLPClassifier(nn.Module):
    def __init__(
        self,
        num_classes: int = 10,
        input_shape: tuple[int, int, int] = (3, 32, 32),
        hidden_dim: int = 512,
        num_hidden_layers: int = 2,
        dropout: float = 0.0, # keeping at 0 for clean comparisons
    ) -> None:
        super().__init__()
        
        c, h, w = input_shape
        in_dim = c * h * w
        
        layers = [nn.Flatten()]
        
        layers.append(nn.Linear(in_dim, hidden_dim))
        layers.append(nn.ReLU(inplace=True))
        if dropout and dropout > 0:
            layers.append(nn.Dropout(p=dropout))
            
        for _ in range(num_hidden_layers - 1):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.ReLU(inplace=True))
            if dropout and dropout > 0:
                layers.append(nn.Dropout(p=dropout))
                
        layers.append(nn.Linear(hidden_dim, num_classes))
        
        self.net = nn.Sequential(*layers)
        self._init_weights()
        
    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
                    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)