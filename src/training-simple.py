import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score
from torch.utils.data import DataLoader
from dataset import N_CLASSES, CustomDataset
from constants import *

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
TRAIN_IMG_PATH = ROOT_DIR / "outputs" / "training.png"
CHECKPOINT_PATH = ROOT_DIR / "outputs" / "drop_detector.pt"

class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.inp = nn.Linear(128, 128) # Convert mel bins to 128 features
        self.pos = nn.Parameter(torch.zeros(1, 300, 128))
        layer = nn.TransformerEncoderLayer(d_model=128, nhead=8, batch_first=True, norm_first=True)
        self.trans = nn.TransformerEncoder(layer, num_layers=6) # Transformer encoder with 6 layers
        self.linear = nn.Linear(128, N_CLASSES) # Fully connected layer

    def forward(self, x):
        x = self.inp(x) + self.pos
        x = self.trans(x)
        return self.linear(x[:, -1])

def evaluate(loader):
    net.eval()
    total_loss = 0
    preds = []
    targets = []
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            out = net(x)
            total_loss += loss_fn(out, y).item()
            preds += out.argmax(1).tolist()
            targets += y.tolist()
    net.train()
    f1 = f1_score(targets, preds, average="macro")
    drop_f1 = f1_score([t > 0 for t in targets], [p > 0 for p in preds])
    return total_loss / len(loader), f1, drop_f1

torch.manual_seed(42)
train_ds = CustomDataset("train", stride=10) # keep 1 in 10 "no drop" windows
val_ds = CustomDataset("val", stride=10)
test_ds = CustomDataset("test", stride=1) # every window, like a real song

train_loader = DataLoader(train_ds, batch_size=64, shuffle=True)
val_loader = DataLoader(val_ds, batch_size=64)
test_loader = DataLoader(test_ds, batch_size=64)

net = NeuralNetwork().to(device) # alt dette er taget fra slides uge 2
optimizer = torch.optim.AdamW(net.parameters(), lr=1e-4)
loss_fn = nn.CrossEntropyLoss(weight=train_ds.class_weights().to(device))

train_losses = []
val_losses = []
best_f1 = -1

for epoch in range(100):
    total_loss = 0
    for x_minibatch, y_minibatch in train_loader:
        x_minibatch, y_minibatch = x_minibatch.to(device), y_minibatch.to(device)
        y_pred = net(x_minibatch)
        loss = loss_fn(y_pred, y_minibatch)

        net.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    train_losses.append(total_loss / len(train_loader))
    val_loss, val_f1, val_drop_f1 = evaluate(val_loader)
    val_losses.append(val_loss)
    print(f"Epoch {epoch}, Loss: {train_losses[-1]:.4f}, Val loss: {val_loss:.4f}, Val F1: {val_f1:.3f}, Drop F1: {val_drop_f1:.3f}")

    if val_f1 > best_f1:
        best_f1 = val_f1
        torch.save(net.state_dict(), CHECKPOINT_PATH)

plt.plot(train_losses, label="train")
plt.plot(val_losses, label="validation")
plt.xlabel("epoch")
plt.ylabel("loss")
plt.legend()
plt.savefig(TRAIN_IMG_PATH)

net.load_state_dict(torch.load(CHECKPOINT_PATH))
test_loss, test_f1, test_drop_f1 = evaluate(test_loader)
print(f"Test Loss: {test_loss:.4f}, Test F1: {test_f1:.3f}, Test Drop F1: {test_drop_f1:.3f}")