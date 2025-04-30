"""
Train a CNN on the C-elegans worm / no-worm dataset (101×101 grayscale).

Outputs into <models/> and <outputs/>:
    • models/worm_cnn.pt               – torch state-dict
    • outputs/training_loss.png
    • outputs/training_accuracy.png
    • outputs/confusion_matrix.png
    • outputs/training_report.docx  (falls back to .txt if python-docx missing)
"""

import os, time, datetime, json, math, itertools
import numpy as np
import torch, torch.nn as nn, torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms
import cv2, matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
from tqdm import tqdm

# ---------- config ----------
IMAGE_SIZE   = 101               # original size of images
BATCH        = 64
EPOCHS       = 30
LR           = 1e-3
VAL_SPLIT    = 0.2
RANDOM_SEED  = 42
ROOT         = os.path.dirname(os.path.abspath(__file__))
DATA_ROOT    = os.path.join(ROOT, "data", "Celegans_ModelGen")
MODELS_DIR   = os.path.join(ROOT, "models")
OUTPUTS_DIR  = os.path.join(ROOT, "outputs")
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(OUTPUTS_DIR, exist_ok=True)
DEVICE       = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ---------- dataset ----------
class WormDataset(Dataset):
    def __init__(self, root):
        self.samples = []
        for lbl in (0,1):
            folder = os.path.join(root, str(lbl))
            imgs   = [os.path.join(folder, f) for f in os.listdir(folder)
                      if f.endswith(".png")]
            self.samples += [(p,lbl) for p in imgs]
        self.tf = transforms.ToTensor()    # converts to C×H×W float[0,1]

    def __len__(self):  return len(self.samples)

    def __getitem__(self, idx):
        p, lbl = self.samples[idx]
        g = cv2.imread(p, cv2.IMREAD_GRAYSCALE)
        g = cv2.equalizeHist(g)            # contrast equalisation
        g = self.tf(g)                     # shape 1×101×101
        return g, torch.tensor(lbl, dtype=torch.long)

ds_full = WormDataset(DATA_ROOT)
n_val   = int(len(ds_full)*VAL_SPLIT)
n_train = len(ds_full)-n_val
train_set, val_set = random_split(ds_full, [n_train, n_val],
                                  generator=torch.Generator().manual_seed(RANDOM_SEED))
ld_train = DataLoader(train_set, batch_size=BATCH, shuffle=True,  num_workers=0)
ld_val   = DataLoader(val_set,   batch_size=BATCH, shuffle=False, num_workers=0)

# ---------- model ----------
class SmallCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1,16,3,padding=1), nn.ReLU(), nn.MaxPool2d(2),   # 16×50×50
            nn.Conv2d(16,32,3,padding=1), nn.ReLU(), nn.MaxPool2d(2),  # 32×25×25
            nn.Conv2d(32,64,3,padding=1), nn.ReLU(), nn.MaxPool2d(2),  # 64×12×12
            nn.Flatten(),
            nn.Linear(64*12*12, 128), nn.ReLU(),
            nn.Linear(128, 2)
        )
    def forward(self,x): return self.net(x)

model = SmallCNN().to(DEVICE)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LR)

# ---------- training loop ----------
train_loss, val_loss = [], []
train_acc,  val_acc  = [], []

t_start = time.time()
for epoch in range(1, EPOCHS+1):
    # train
    model.train(); tl=0; correct=0; total=0
    for x,y in ld_train:
        x,y = x.to(DEVICE), y.to(DEVICE)
        optimizer.zero_grad()
        out = model(x)
        loss = criterion(out,y)
        loss.backward(); optimizer.step()
        tl += loss.item()*y.size(0)
        pred = out.argmax(1); correct += (pred==y).sum().item()
        total += y.size(0)
    train_loss.append(tl/total); train_acc.append(correct/total)

    # val
    model.eval(); vl=0; correct=0; total=0
    with torch.no_grad():
        for x,y in ld_val:
            x,y = x.to(DEVICE), y.to(DEVICE)
            out = model(x); loss=criterion(out,y)
            vl += loss.item()*y.size(0)
            pred=out.argmax(1); correct+=(pred==y).sum().item()
            total+=y.size(0)
    val_loss.append(vl/total); val_acc.append(correct/total)

    print(f"Epoch {epoch:>2}/{EPOCHS}  "
          f"train-acc {train_acc[-1]:.3f}  val-acc {val_acc[-1]:.3f}")

train_time = time.time()-t_start

# ---------- evaluation on full val set ----------
model.eval(); y_true=[]; y_pred=[]
with torch.no_grad():
    for x,y in ld_val:
        x=x.to(DEVICE)
        out = model(x).cpu()
        y_true += y.tolist()
        y_pred += out.argmax(1).tolist()

acc = accuracy_score(y_true, y_pred)
cm  = confusion_matrix(y_true, y_pred)
report = classification_report(y_true, y_pred, digits=4)
print("\nValidation accuracy:", acc)
print(cm)
print(report)

# ---------- save model ----------
model_path = os.path.join(MODELS_DIR, "worm_cnn.pt")
torch.save({
    "model_state": model.state_dict(),
    "input_size":  (1, IMAGE_SIZE, IMAGE_SIZE)
}, model_path)
print("Model saved to", model_path)

# ---------- plots ----------
e=np.arange(1,EPOCHS+1)
plt.figure(); plt.plot(e, train_loss, label="train"); plt.plot(e, val_loss,label="val")
plt.title("Loss"); plt.xlabel("epoch"); plt.legend()
plt.savefig(os.path.join(OUTPUTS_DIR,"training_loss.png")); plt.close()

plt.figure(); plt.plot(e, train_acc, label="train"); plt.plot(e, val_acc, label="val")
plt.title("Accuracy"); plt.xlabel("epoch"); plt.legend()
plt.savefig(os.path.join(OUTPUTS_DIR,"training_accuracy.png")); plt.close()

plt.figure(); plt.imshow(cm, cmap="Blues"); plt.title("Confusion matrix")
for i,j in itertools.product(range(2),range(2)):
    plt.text(j,i,str(cm[i,j]),ha='center',va='center',color='white')
plt.savefig(os.path.join(OUTPUTS_DIR,"confusion_matrix.png")); plt.close()

# ---------- tiny report ----------
try:
    from docx import Document
    doc = Document(); doc.add_heading("Project 6 – Worm CNN", 0)
    tbl = doc.add_table(rows=0, cols=2); tbl.style="Light List"
    def add(k,v): row=tbl.add_row().cells; row[0].text=k; row[1].text=v
    add("Visual verification","[add your note]")
    add("Split sizes",f"Train {n_train}, Val {n_val}")
    add("Image size","101×101 grayscale")
    add("Preprocess","equalizeHist → ToTensor")
    add("Model params", f"{sum(p.numel() for p in model.parameters()):,}")
    add("Optimizer","Adam  lr=1e-3  epochs=30")
    add("Training time", f"{train_time:.1f}s on {DEVICE}")
    add("Val accuracy", f"{acc:.4f}")
    add("Confusion matrix", np.array2string(cm))
    add("Classification report", report)
    doc_path=os.path.join(OUTPUTS_DIR,"training_report.docx")
    doc.save(doc_path)
except ImportError:
    with open(os.path.join(OUTPUTS_DIR,"training_report.txt"),"w") as f:
        f.write(report)
        doc_path="training_report.txt"

print("Report saved to", doc_path)
