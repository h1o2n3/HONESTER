
import torch

class MultiHotLayer(torch.nn.Module):
    def __init__(self, dim1,outdim):
        super().__init__()
        self.fc1 = torch.nn.Linear(dim1, outdim*2)
        self.fc2 = torch.nn.Linear(outdim*2, outdim)
        # self.act = torch.nn.ReLU()

        torch.nn.init.xavier_normal_(self.fc1.weight)
        torch.nn.init.xavier_normal_(self.fc2.weight)
        
    def forward(self, x1):
        # if len(x1.size())!= 3:
        #     for i in range(x1.size()[0]):
        #         for j in range(x1.size()[1]):
                    
        h = self.fc1(x1)
        c = self.fc2(h)
        # print(c.size())
        return c

