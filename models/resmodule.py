from torch import nn 



class Residual(nn.Module):
    def __init__(self,in_features, num_layers=10, dropout=0.4):
        super().__init__()
        act = nn.PReLU
        norm = nn.BatchNorm1d
        self.layers = nn.ModuleList()
        for i in range(num_layers):
            self.layers.append(nn.Sequential(
                nn.Dropout(dropout),
                nn.Linear(in_features, in_features),
                act(),
                norm(in_features)
            ))
            self.layers.append(norm(in_features))

    def forward(self, x):
        for layer, norm in zip(self.layers[::2],self.layers[1::2]):
            residual = x.clone()
            x = layer(x)
            x = x + residual
            x = norm(x)
        return x
    
    
    