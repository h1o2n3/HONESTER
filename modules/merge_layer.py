
import torch


class MergeLayer(torch.nn.Module):
    def __init__(self, dim1, dim2, dim3, dim4):
        super().__init__()
        
        
        self.fc1 = torch.nn.Linear(dim1 + dim2, dim3)
        self.fc2 = torch.nn.Linear(dim3, dim4)
        self.act = torch.nn.ReLU()

        torch.nn.init.xavier_normal_(self.fc1.weight)
        torch.nn.init.xavier_normal_(self.fc2.weight)
        
    def forward(self, x1, x2):
        
        if x1.shape[0] != x2.shape[0]:
            x1 = x1.view([x2.shape[0],-1])
        
        x = torch.cat([x1, x2], dim=1)
        
        
        h = self.act(self.fc1(x))
        return self.fc2(h)

