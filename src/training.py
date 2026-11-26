"""Train the drop detector: is a drop coming within the next 10 seconds?

Usage: python src/training.py [--no-aug]
Uses every GPU: the script starts itself once per GPU (with torchrun), and each
copy trains on its share of every batch. The songs are kept on the GPUs, so the
CPU does almost nothing.

The model looks at the last 60 s of a song and says how far away the next drop is
(11 classes: no drop within 10 s, or which second it is in). We use it as a
warning: warn when P(no drop) is below a threshold, and keep the warning on for 2 s.
The threshold is picked on the validation songs: warn about as many drops as
possible with at most 15 false alarms per hour.

Augmentation: every epoch trains on each original song plus one randomly picked
precomputed version of it (pitch, speed, compression, telephone or background
noise, from songs_10fps_augmented.pkl), and every batch gets a random loudness
change, a little noise and masked-out bits (dataset.augment). --no-aug turns
all of that off (original songs only), as a baseline; that run saves to
outputs/drop_detector_rune_noaug.pt and outputs/training_noaug.png instead.

Saves the best model to outputs/drop_detector_rune.pt and the loss curves to outputs/training.png,
then prints how well it warns about the drops in the test songs.
"""
import contextlib
import os
import sys

import matplotlib
import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel

from constants import FPS, ROOT_DIR
from dataset import N_CLASSES, CustomDataset, augment, windows

# Start one copy of this script per GPU. torchrun gives each copy its GPU number.
if "LOCAL_RANK" not in os.environ:
    os.execvp(sys.executable, [sys.executable, "-m", "torch.distributed.run", "--standalone",
                               f"--nproc-per-node={torch.cuda.device_count()}", *sys.argv])

matplotlib.use("Agg")
import matplotlib.pyplot as plt

AUGMENT = "--no-aug" not in sys.argv
SUFFIX = "" if AUGMENT else "_noaug"
CHECKPOINT_PATH = ROOT_DIR / "outputs" / f"drop_detector_rune{SUFFIX}.pt"
TRAIN_IMG_PATH = ROOT_DIR / "outputs" / f"training{SUFFIX}.png"
EPOCHS = 20
WINDOW = 600 # frames the model sees (60 s)
BATCH = 512 # windows per training step, split over the GPUs
CHUNK = 128 # windows per forward/backward pass on one GPU (memory); gradients are summed up to BATCH
LR = 4e-5
STRIDE = 3 # training keeps 1 in 3 "no drop" windows (all windows near a drop are kept)
HOLD = 2 * FPS # keep a warning on for 2 s (20 frames), so it does not flicker on and off
MAX_FALSE_ALARMS = 15 # per hour, on the validation songs
THRESHOLDS = [0.0, *np.geomspace(1e-4, 0.99, 40)] # tried each epoch; 0 = never warn

dist.init_process_group("nccl")
rank = dist.get_rank() # which GPU this copy uses: 0, 1, 2 or 3
world = dist.get_world_size() # number of GPUs
device = torch.device("cuda", rank)
torch.cuda.set_device(device)
main = rank == 0 # only the copy on GPU 0 prints and saves files


class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.inp = nn.Linear(128, 128) # Convert mel bins to 128 features
        self.pos = nn.Parameter(torch.zeros(1, WINDOW, 128)) # a learned position for each of the 600 frames (60 s)
        layer = nn.TransformerEncoderLayer(d_model=128, nhead=8, dim_feedforward=512, dropout=0.3,
                                           batch_first=True, norm_first=True)
        self.trans = nn.TransformerEncoder(layer, num_layers=1, enable_nested_tensor=False) # one attention layer
        self.linear = nn.Linear(128, N_CLASSES) # Fully connected layer

    def forward(self, x):
        x = self.inp(x) + self.pos
        x = self.trans(x)
        return self.linear(x[:, -1]) # predict from the newest frame


def batches(data, shuffle):
    """This GPU's share of the windows, as (x, y) minibatches. Every GPU gets
    the same number of batches (DistributedDataParallel needs that)."""
    frames, starts, labels = data
    if shuffle:
        order = torch.randperm(len(starts), device=device) # same order on every GPU: same seed
    else:
        order = torch.arange(len(starts), device=device)
    mine = order[rank::world][:len(order) // world] # every world-th window, starting at this GPU's number
    for i in range(0, len(mine), BATCH // world):
        b = mine[i:i + BATCH // world]
        yield windows(frames, starts[b], WINDOW), labels[b]


def evaluate(data):
    """Mean loss, and P(no drop) for every window of a split, in order.
    Each GPU does one quarter of the windows."""
    frames, starts, labels = data
    quarter = (len(starts) + world - 1) // world
    first, last = rank * quarter, min((rank + 1) * quarter, len(starts))
    model.eval()
    total, p_no_drop = 0.0, []
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16):
        for i in range(first, last, CHUNK * 2):
            j = min(i + CHUNK * 2, last)
            out = model(windows(frames, starts[i:j], WINDOW)).float()
            total += loss_fn(out, labels[i:j]).item() * (j - i)
            p_no_drop.append(out.softmax(1)[:, 0].cpu().numpy())
    model.train()
    shares = [None] * world
    dist.all_gather_object(shares, (total, np.concatenate(p_no_drop))) # put the GPUs' shares together
    return sum(s[0] for s in shares) / len(starts), np.concatenate([s[1] for s in shares])


def hold(warn, songs):
    """Keep each warning on for HOLD more frames (but not into the next song)."""
    held = warn.copy()
    for k in range(1, HOLD + 1):
        held[k:] |= warn[:-k] & (songs[k:] == songs[:-k])
    return held


def runs(mask):
    """Where each stretch of True values in mask starts and ends: [(start, end), ...]."""
    change = np.diff(np.concatenate([[0], mask.astype(int), [0]]))
    return zip(np.flatnonzero(change == 1), np.flatnonzero(change == -1))


def alarm_counts(warn, drop, songs):
    """How the warnings work as an alarm. A drop counts as warned if the warning is on
    at some point in the 10 s before it. Every stretch of time the warning is on while
    no drop is coming within 10 s is a false alarm - also when the same warning later
    runs on into a drop, so warning all the time does not pay. The HOLD frames right
    after a drop are allowed, since a held warning always runs on past its drop.
    Returns (drops warned, drops, seconds of warning for each warned drop, false alarms)."""
    allowed = hold(drop, songs) # a drop is coming, or one just happened
    warned, ahead, false_alarms, n_drops = 0, [], 0, 0
    for s in np.unique(songs):
        w, d, ok = warn[songs == s], drop[songs == s], allowed[songs == s]
        for a, b in runs(d): # the 10 s before one drop; window b - 1 ends right on the drop
            n_drops += 1
            on = np.flatnonzero(w[a:b])
            if len(on) > 0:
                warned += 1
                ahead.append((b - 1 - a - on[0]) / FPS)
        false_alarms += len(list(runs(w & ~ok))) # warning on with no drop coming
    return warned, n_drops, ahead, false_alarms


def alarm_report(warn, drop, songs):
    """(share of drops warned, median seconds of warning, false alarms per hour); see alarm_counts."""
    warned, n_drops, ahead, false_alarms = alarm_counts(warn, drop, songs)
    hours = len(warn) / FPS / 3600
    return warned / n_drops, np.median(ahead) if ahead else 0.0, false_alarms / hours


def f1(precision, recall):
    return 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0.0


def f1_scores(p_no_drop, threshold, drop, songs):
    """Two F1 scores at one threshold.
    Event F1: recall = share of drops warned; precision = warned drops / (warned drops +
    false alarms), i.e. the share of alarms that were followed by a real drop.
    Window F1: every window on its own, without the 2 s hold: is P(no drop) < threshold
    right about "a drop is coming within 10 s"?"""
    warned, n_drops, _, false_alarms = alarm_counts(hold(p_no_drop < threshold, songs), drop, songs)
    event = f1(warned / max(warned + false_alarms, 1), warned / n_drops)
    pred = p_no_drop < threshold
    hits = (pred & drop).sum()
    window = f1(hits / max(pred.sum(), 1), hits / max(drop.sum(), 1))
    return event, window


def choose_threshold(p_no_drop, drop, songs):
    """The threshold that warns about the most drops with at most MAX_FALSE_ALARMS per hour.
    Returns (threshold, (share of drops warned, median seconds of warning, false alarms per hour))."""
    results = [(th, alarm_report(hold(p_no_drop < th, songs), drop, songs)) for th in THRESHOLDS]
    allowed = [r for r in results if r[1][2] <= MAX_FALSE_ALARMS]
    return max(allowed, key=lambda r: r[1][0])


torch.manual_seed(42) # same start weights on every GPU
# Training holds every augmented version on the GPU; each epoch picks which ones to use.
train_ds = CustomDataset("train", stride=STRIDE, window=WINDOW, augmentation="all" if AUGMENT else False)
train_all = train_ds.device_tensors(device)
# Validation and test use every window of the original songs, like a real song: few windows are near a drop.
val_ds = CustomDataset("val", stride=1, window=WINDOW)
test_ds = CustomDataset("test", stride=1, window=WINDOW)
val = val_ds.device_tensors(device)
test = test_ds.device_tensors(device)
val_drop = val[2].cpu().numpy() > 0 # is a drop coming within 10 s?
test_drop = test[2].cpu().numpy() > 0
val_songs = val_ds.index[:, 0] # which song each window is from
test_songs = test_ds.index[:, 0]
if main:
    print(f"augmentation {'on' if AUGMENT else 'off'}: train {len(train_ds)} windows in {len(train_ds.mels)} song versions"
          f"{' (original + 1 of 7 augmented per song used each epoch)' if AUGMENT else ''}, "
          f"val {len(val[2])}, test {len(test[2])} windows on {world} GPUs", flush=True)

model = NeuralNetwork().to(device) # alt dette er taget fra slides uge 2
net = DistributedDataParallel(model) # averages the gradients over the GPUs
optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.1)
scaler = torch.amp.GradScaler() # fp16 needs the loss scaled up so small gradients don't round to 0
loss_fn = nn.CrossEntropyLoss()

train_losses, val_losses, best_warned, threshold = [], [], -1.0, None
for epoch in range(EPOCHS):
    torch.manual_seed(epoch) # so every GPU shuffles the same way
    # This epoch: every original song plus one random augmented version of it (same pick on every GPU).
    if AUGMENT:
        keep = torch.from_numpy(train_ds.pick_augmentations(np.random.default_rng(epoch))).to(device)
        train = (train_all[0], train_all[1][keep], train_all[2][keep])
    else:
        train = train_all
    running = torch.zeros((), device=device)
    for x_minibatch, y_minibatch in batches(train, shuffle=True):
        if AUGMENT:
            x_minibatch = augment(x_minibatch) # random loudness and masked-out bits, so the model can't memorize songs
        # The minibatch goes through in chunks of CHUNK windows to fit in GPU memory; the
        # gradients add up, so the step is the same as for the whole minibatch at once.
        net.zero_grad()
        n = len(y_minibatch)
        for k in range(0, n, CHUNK):
            x, y = x_minibatch[k:k + CHUNK], y_minibatch[k:k + CHUNK]
            last = k + CHUNK >= n
            with (contextlib.nullcontext() if last else net.no_sync()): # sync gradients over the GPUs once, at the end
                with torch.autocast("cuda", dtype=torch.float16): # fp16: about twice as fast
                    loss = loss_fn(net(x), y) * (len(y) / n) # this chunk's share of the minibatch mean
                scaler.scale(loss).backward()
            running += loss.detach() * n # no .item() here: it would make the CPU wait for the GPU
        scaler.step(optimizer)
        scaler.update()

    dist.all_reduce(running) # add up the GPUs
    train_losses.append(running.item() / len(train[2]))
    val_loss, p_no_drop = evaluate(val)
    val_losses.append(val_loss)
    if main:
        epoch_threshold, (warned, ahead, false_alarms) = choose_threshold(p_no_drop, val_drop, val_songs)
        if warned > best_warned:
            best_warned, threshold = warned, epoch_threshold
            torch.save(model.state_dict(), CHECKPOINT_PATH)
        event_f1, window_f1 = f1_scores(p_no_drop, epoch_threshold, val_drop, val_songs)
        print(f"Epoch {epoch}, Loss: {train_losses[-1]:.4f}, Val loss: {val_loss:.4f}, Val: {warned:.0%} of drops warned, "
              f"median {ahead:.1f} s ahead, {false_alarms:.1f} false alarms per hour, threshold {epoch_threshold:.2g}, "
              f"event F1 {event_f1:.3f}, window F1 {window_f1:.3f}", flush=True)

if main:
    plt.plot(train_losses, label="train")
    plt.plot(val_losses, label="validation")
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.legend()
    plt.savefig(TRAIN_IMG_PATH, dpi=120)

dist.barrier() # wait until GPU 0 has saved the best model
model.load_state_dict(torch.load(CHECKPOINT_PATH, map_location=device))
test_loss, p_no_drop = evaluate(test)
if main:
    warn = hold(p_no_drop < threshold, test_songs)
    warned, ahead, false_alarms = alarm_report(warn, test_drop, test_songs)
    event_f1, window_f1 = f1_scores(p_no_drop, threshold, test_drop, test_songs)
    print(f"Test loss {test_loss:.4f}, threshold {threshold:.2g}: {warned:.0%} of drops warned, "
          f"median {ahead:.1f} s ahead, {false_alarms:.1f} false alarms per hour, "
          f"event F1 {event_f1:.3f}, window F1 {window_f1:.3f}")
dist.destroy_process_group()