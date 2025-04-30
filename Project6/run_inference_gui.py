"""
Project-6 inference GUI
-----------------------
• Loads the SmallCNN checkpoint  (best_model.pt)
• Lets the user pick individual .png images OR a directory of them
• Writes results to  outputs/prediction_elegans.xlsx  (creates folder if needed)
"""

import os, sys, json, cv2, time, numpy as np, pandas as pd, tkinter as tk
from tkinter import filedialog, messagebox
import torch, torch.nn as nn

# ───────────────────────── 1.  network definition ──────────────────────────
class SmallCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1,16,3,padding=1), nn.BatchNorm2d(16), nn.ReLU(),
            nn.MaxPool2d(2), nn.Dropout2d(0.2),

            nn.Conv2d(16,32,3,padding=1), nn.BatchNorm2d(32), nn.ReLU(),
            nn.MaxPool2d(2), nn.Dropout2d(0.3),

            nn.Conv2d(32,64,3,padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.MaxPool2d(2), nn.Dropout2d(0.4),

            nn.Flatten(),                                 # 64×12×12 (96×96 input)
            nn.Linear(64*12*12,128), nn.ReLU(), nn.Dropout(0.5),
            nn.Linear(128,2)
        )
    def forward(self,x): return self.net(x)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ───────────────────────── 2.  pre-processing  ─────────────────────────────
def preprocess(gray):
    clahe = cv2.createCLAHE(2.0,(8,8)); gray = clahe.apply(gray)
    gray  = cv2.medianBlur(gray,3)
    edges = cv2.Canny(gray,50,150)
    return cv2.addWeighted(gray,0.8,edges,0.2,0)

def img_to_tensor(gray):
    g = preprocess(gray).astype(np.float32)/255.0   # (H,W) 0-1
    return torch.from_numpy(g[None,None,:,:])       # (1,1,H,W)

# ───────────────────────── 3.  single-image inference ──────────────────────
@torch.inference_mode()
def predict(model, img_path):
    g = cv2.imread(img_path,cv2.IMREAD_GRAYSCALE)
    if g is None: raise ValueError(f"Cannot read {img_path}")
    x = img_to_tensor(g).to(DEVICE)
    logits  = model(x)
    prob    = torch.softmax(logits,1)[0]
    label   = int(prob.argmax())
    return label, float(prob[label])

# optional overlay preview
def overlay(img_path, lab, prob):
    img = cv2.imread(img_path)
    txt = f"Pred: {lab} ({prob*100:.1f}%)"
    cv2.putText(img,txt,(10,28),cv2.FONT_HERSHEY_SIMPLEX,0.7,(0,255,0),2)
    cv2.imshow("Prediction",img); cv2.waitKey(1)

# ───────────────────────── 4.  GUI  ────────────────────────────────────────
class InferenceGUI(tk.Tk):
    def __init__(self, model):
        super().__init__()
        self.model=model
        self.title("Project-6 inference"); self.geometry("400x220")
        tk.Label(self,text="Select inference method",font=("Arial",14)).pack(pady=20)
        tk.Button(self,text="Individual images",font=("Arial",12),
                  command=self.pick_imgs).pack(pady=5,fill=tk.X,padx=60)
        tk.Button(self,text="Directory of images",font=("Arial",12),
                  command=self.pick_dir).pack(pady=5,fill=tk.X,padx=60)
        tk.Button(self,text="Quit",font=("Arial",12),command=self.destroy)\
            .pack(pady=15)

    def pick_imgs(self):
        files=filedialog.askopenfilenames(filetypes=[("PNG","*.png")])
        if files: self.run(list(files))

    def pick_dir(self):
        d=filedialog.askdirectory()
        if not d: return
        files=[os.path.join(d,f) for f in os.listdir(d) if f.lower().endswith(".png")]
        if files: self.run(files)
        else: messagebox.showerror("No PNG","No .png images in that folder.")

    def run(self, paths):
        recs, counts = [], {0:0,1:0}
        for p in paths:
            try:
                lab, prob = predict(self.model,p)
                recs.append((os.path.basename(p),lab)); counts[lab]+=1
                overlay(p,lab,prob)
            except Exception as e:
                print("Error on",p,":",e)

        # Excel  → outputs/prediction_elegans.xlsx
        outputs_dir = os.path.join(os.path.dirname(__file__),"outputs")
        os.makedirs(outputs_dir,exist_ok=True)
        xlsx_path   = os.path.join(outputs_dir,"prediction_elegans.xlsx")

        df = pd.DataFrame(recs,columns=["filename","label"])
        df.loc[len(df)] = ["",""]                # blank line
        for k in (0,1): df.loc[len(df)] = [f"total {k}", counts[k]]
        df.to_excel(xlsx_path,index=False)

        messagebox.showinfo("Done",f"Results written to:\n{xlsx_path}")
        self.destroy()

# ───────────────────────── 5.  entry-point  ────────────────────────────────
def main():
    root=tk.Tk(); root.withdraw()
    mpath=filedialog.askopenfilename(
        title="Select checkpoint (best_model.pt)",
        filetypes=[("PyTorch checkpoint","*.pt *.pth"),("All","*.*")]
    )
    if not mpath: return
    try:
        ck = torch.load(mpath,map_location=DEVICE)
        model = SmallCNN().to(DEVICE)
        model.load_state_dict(ck["model_state"], strict=True)
        model.eval()
    except Exception as e:
        messagebox.showerror("Error",f"Could not load checkpoint:\n{e}")
        return
    InferenceGUI(model).mainloop()

if __name__=="__main__":
    main()
