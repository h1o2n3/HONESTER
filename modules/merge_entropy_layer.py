
import torch

class MergeEntropyLayer(torch.nn.Module):
    def __init__(self, dim1):
        super().__init__()
        self.fc1 = torch.nn.Linear(dim1*3, dim1*2)
        self.fc2 = torch.nn.Linear(dim1*2, 1)
        self.act = torch.nn.ReLU()

        torch.nn.init.xavier_normal_(self.fc1.weight)
        torch.nn.init.xavier_normal_(self.fc2.weight)
        
    def forward(self, x1, x2,entropy):

        x = torch.cat([x1, x2,entropy], dim=1)
        h = self.act(self.fc1(x))
        return self.fc2(h)

