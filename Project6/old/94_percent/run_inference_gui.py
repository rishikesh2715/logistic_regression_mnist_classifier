"""
GUI launcher for Project-6 worm / no-worm CNN inference.

Assumptions
-----------
project_root/
├─ inference_scripts/          (not used by this file)
├─ models/
│    └─ worm_cnn.pt
├─ outputs/
└─ run_inference_gui.py        ← THIS SCRIPT
"""

import os, cv2, numpy as np, pandas as pd, torch, torch.nn as nn
import tkinter as tk
from tkinter import filedialog, messagebox
from skimage.feature import hog, local_binary_pattern

# ---------- basic preprocess (same as training) ----------
def preprocess(img):
    img = cv2.equalizeHist(img)              # simple contrast stretch
    return img                                # keep 101×101

# ---------- tiny CNN architecture (matches training) ----------
def build_cnn():
    class SmallCNN(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.net = torch.nn.Sequential(
                torch.nn.Conv2d(1,16,3,padding=1),  torch.nn.ReLU(), torch.nn.MaxPool2d(2),
                torch.nn.Conv2d(16,32,3,padding=1), torch.nn.ReLU(), torch.nn.MaxPool2d(2),
                torch.nn.Conv2d(32,64,3,padding=1), torch.nn.ReLU(), torch.nn.MaxPool2d(2),
                torch.nn.Flatten(),
                torch.nn.Linear(64*12*12,128), torch.nn.ReLU(),
                torch.nn.Linear(128,2)
            )
        def forward(self, x):            # just delegate
            return self.net(x)
    return SmallCNN()

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ---------- single-image inference ----------
def img_to_tensor(path):
    g = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if g is None: raise ValueError(path)
    g = preprocess(g).astype(np.float32) / 255.0
    t = torch.from_numpy(g).unsqueeze(0).unsqueeze(0)  # 1×1×H×W
    return t.to(DEVICE)

def predict(model, tensor):
    with torch.no_grad():
        out = model(tensor)
        prob = torch.softmax(out,1)[0]
    lab = int(prob.argmax()); return lab, float(prob[lab])

# ---------- GUI workflow ----------
def main():
    root = tk.Tk(); root.withdraw()

    # 1. pick model
    mpath = filedialog.askopenfilename(
        title="Select trained CNN model (worm_cnn.pt)",
        filetypes=[("PyTorch model","*.pt")])
    if not mpath: return

    # 2. pick folder of test images
    folder = filedialog.askdirectory(title="Select folder of PNG test images")
    if not folder: return
    imgs = [os.path.join(folder,f) for f in os.listdir(folder) if f.endswith(".png")]
    if not imgs:
        messagebox.showerror("No PNG","Folder contains no .png files."); return

    # 3. load model
    model = build_cnn(); model.load_state_dict(torch.load(mpath,map_location=DEVICE)["model_state"])
    model.eval().to(DEVICE)

    # 4. run inference
    rows=[]; cnt={0:0,1:0}
    for p in imgs:
        try:
            lab,_ = predict(model,img_to_tensor(p))
            rows.append((os.path.basename(p), lab)); cnt[lab]+=1
        except Exception as e:
            print("err",p,e)

    # 5. to Excel -> outputs/prediction_elegans.xlsx
    df = pd.DataFrame(rows, columns=["filename","label"])
    df.loc[len(df)] = ["",""]
    for k in (0,1): df.loc[len(df)] = [f"total {k}", cnt[k]]

    proj_root = os.path.dirname(os.path.abspath(__file__))
    out_dir   = os.path.join(proj_root, "outputs")
    os.makedirs(out_dir, exist_ok=True)
    out_path  = os.path.join(out_dir, "prediction_elegans.xlsx")
    df.to_excel(out_path, index=False)

    messagebox.showinfo("Done", f"Excel saved to:\n{out_path}")
    print("Excel saved to:", out_path)

if __name__ == "__main__":
    main()
