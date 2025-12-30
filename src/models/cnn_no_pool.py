import torch
import torch.nn as nn

class CNNNoPool(nn.Module):
    def __init__(
        self,
        num_classes: int = 10, 
        channels: tuple[int, int] = (32, 64),
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        
        c1, c2 = channels
        
        self.features = nn.Sequential(
            nn.Conv2d(3, c1, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            
            nn.Conv2d(c1, c2, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
        )
        
        feat_dim = c2 * 32 * 32
        
        head = [
            nn.Flatten(),
            nn.Linear(feat_dim, 256),
            nn.ReLU(inplace=True),
        ]
        
        if dropout and dropout > 0:
            head.append(nn.Dropout(p=dropout))
            
        head.append(nn.Linear(256, num_classes))
        
        self.classifier = nn.Sequential(*head)
        self._init_weights()
        
    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
                    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.classifier(x)
        return x