import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score
from torch.utils.data import DataLoader

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.inp = nn.Linear(128, 128) # Convert mel bins to 128 features
        self.pos = nn.Parameter(torch.zeros(1, 300, 128))
        layer = nn.TransformerEncoderLayer(d_model=128, nhead=8, batch_first=True, norm_first=True)
        self.trans = nn.TransformerEncoder(layer, num_layers=6) # Transformer encoder with 6 layers
        self.linear = nn.Linear(128, 11) # Fully connected layer

    def forward(self, x):
        x = self.inp(x) + self.pos
        x = self.trans(x)
        return self.linear(x[:, -1])

torch.manual_seed(42)
net = NeuralNetwork().to(device) # alt dette er taget fra slides uge 2
optimizer = torch.optim.AdamW(net.parameters(), lr=1e-5) # Bare default lr, betas, eps og weight_decay
loss_fn = nn.CrossEntropyLoss()

for epoch in range(100):
    for x_minibatch, y_minibatch in waveformLoader:
        x_minibatch, y_minibatch = x_minibatch.to(device), y_minibatch.to(device)
        y_pred = net(x_minibatch)
        loss = loss_fn(y_pred, y_minibatch)

        net.zero_grad()
        loss.backward()
        optimizer.step()
        
    if epoch % 10 == 0:
        print(f"Epoch {epoch}, Loss: {loss.item()}")

with torch.no_grad():
    pred = net(testX.to(device))[0].cpu()
    testLoss = loss_fn(pred, testy)
print(f"Test Loss: {testLoss.item()}")