import torch
import torch.nn as nn
from .locally_connected import LocallyConnected2d

class CNNLC1(nn.Module):
    def __init__(self,
                 num_classes=10,
                 channels=(32, 64),
                 dropout=0.0):
        super().__init__()
        
        c1, c2 = channels
        
        self.lc1 = LocallyConnected2d(3, c1, output_size=(32, 32), kernel_size=3, stride=1, padding=1)
        
        self.features = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2), 
            
            nn.Conv2d(c1, c2, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )
        
        feat_dim = c2 * 8 * 8
        head = [nn.Flatten(), nn.Linear(feat_dim, 256), nn.ReLU(inplace=True)]
        
        if dropout and dropout > 0:
            head.append(nn.Dropout(p=dropout))
        
        head.append(nn.Linear(256, num_classes))
        self.classifier = nn.Sequential(*head)
        
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
        x = self.lc1(x)
        x = self.features(x)
        x = self.classifier(x)
        return x