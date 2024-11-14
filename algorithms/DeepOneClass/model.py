import torch 
import torch.nn as nn 

class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        
        self.b1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=8, kernel_size=5, padding=2, bias=False),
            nn.BatchNorm2d(8, eps=1e-4, affine=False),
            nn.LeakyReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        self.b2 = nn.Sequential(
            nn.Conv2d(in_channels=8, out_channels=4, kernel_size=5, padding=2, bias=False),
            nn.BatchNorm2d(4, eps=1e-4, affine=False),
            nn.LeakyReLU(),
            nn.MaxPool2d(kernel_size=2)
        )

        self.out = nn.Sequential(
            nn.Linear(in_features=4*7*7, out_features=32, bias=False),
        )

    def forward(self, x):
        out1 = self.b1(x)
        out2 = self.b2(out1)
        out2 = out2.flatten(start_dim=1)
        out = self.out(out2)
        return out
    
if __name__=="__main__":

    DEVICE="mps"
    x = torch.randn(10, 1, 28, 28).to(DEVICE)
    model = CNN().to(DEVICE)
    out = model(x)
    print(out.shape)