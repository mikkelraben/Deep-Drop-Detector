"""Train the drop detector on songs_10fps.pkl (build it with build_features.py).

Usage: python training.py [epochs=100]
Saves the weights with the best validation F1 to drop_detector.pt and the
loss curves to training.png, then reports F1 on the test songs.
"""
import sys

import matplotlib
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import f1_score
from torch.utils.data import DataLoader

from dataset import N_CLASSES, CustomDataset

matplotlib.use("Agg")
import matplotlib.pyplot as plt

EPOCHS = int(sys.argv[1]) if len(sys.argv) > 1 else 100
BATCH = 64
STRIDE = 10   # keep 1 in 10 "no drop" windows when training (all drop windows are kept)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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
    """Mean loss, macro F1 over the 11 classes, and F1 of "drop within 5 s or not"."""
    net.eval()
    total, n, preds, targets = 0.0, 0, [], []
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            out = net(x)
            total += loss_fn(out, y).item() * len(y)
            n += len(y)
            preds.append(out.argmax(1).cpu())
            targets.append(y.cpu())
    net.train()
    p, t = torch.cat(preds).numpy(), torch.cat(targets).numpy()
    return total / n, f1_score(t, p, average="macro"), f1_score(t > 0, p > 0)


torch.manual_seed(42)
train_ds = CustomDataset("train", stride=STRIDE)
val_ds = CustomDataset("val", stride=STRIDE)
test_ds = CustomDataset("test", stride=1)    # every window, like a real song
def make_loader(ds, shuffle=False):
    return DataLoader(ds, batch_size=BATCH, shuffle=shuffle, num_workers=4,
                      pin_memory=device.type == "cuda")

train_loader = make_loader(train_ds, shuffle=True)
val_loader = make_loader(val_ds)
test_loader = make_loader(test_ds)
print(f"train {len(train_ds)}, val {len(val_ds)}, test {len(test_ds)} windows on {device}")

net = NeuralNetwork().to(device) # alt dette er taget fra slides uge 2
optimizer = torch.optim.AdamW(net.parameters(), lr=1e-4) # Bare default lr, betas, eps og weight_decay
# Weight rare classes up: ~70 % of training windows are "no drop".
loss_fn = nn.CrossEntropyLoss(weight=train_ds.class_weights().to(device))

train_losses, val_losses, best_f1 = [], [], -1.0
for epoch in range(EPOCHS):
    running = 0.0
    for x_minibatch, y_minibatch in train_loader:
        x_minibatch, y_minibatch = x_minibatch.to(device), y_minibatch.to(device)
        y_pred = net(x_minibatch)
        loss = loss_fn(y_pred, y_minibatch)

        net.zero_grad()
        loss.backward()
        optimizer.step()
        running += loss.item() * len(y_minibatch)

    train_losses.append(running / len(train_ds))
    val_loss, val_f1, val_drop_f1 = evaluate(val_loader)
    val_losses.append(val_loss)
    if val_f1 > best_f1:
        best_f1 = val_f1
        torch.save(net.state_dict(), "drop_detector.pt")
    print(f"Epoch {epoch}, Loss: {train_losses[-1]:.4f}, Val loss: {val_loss:.4f}, "
          f"Val F1: {val_f1:.3f}, Val drop/no-drop F1: {val_drop_f1:.3f}")

plt.plot(train_losses, label="train")
plt.plot(val_losses, label="validation")
plt.xlabel("epoch")
plt.ylabel("loss")
plt.legend()
plt.savefig("training.png", dpi=120)

net.load_state_dict(torch.load("drop_detector.pt"))
testLoss, test_f1, test_drop_f1 = evaluate(test_loader)
print(f"Test Loss: {testLoss:.4f}, Test F1: {test_f1:.3f}, Test drop/no-drop F1: {test_drop_f1:.3f}")
