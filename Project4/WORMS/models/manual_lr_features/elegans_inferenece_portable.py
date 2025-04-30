"""
Portable inference for worm / no‑worm.

Requirements:  numpy  opencv-python  pandas  openpyxl
"""

import os, cv2, json, numpy as np, pandas as pd, tkinter as tk
from tkinter import filedialog, messagebox
from skimage.feature import hog, local_binary_pattern

# ---------- load NP weights ----------
def load_np_model(npz_path):
    d = np.load(npz_path)
    W, b = d["W"], float(d["b"])
    return W, b

def sigmoid(z): return 1/(1+np.exp(-z))

def predict_batch(feat, W, b):
    p = sigmoid(feat.dot(W)+b)
    return (p>=0.5).astype(np.int8), p

# ---------- features ----------
def preprocess(img):
    clahe=cv2.createCLAHE(2.0,(8,8)); img=clahe.apply(img)
    img=cv2.medianBlur(img,3); edges=cv2.Canny(img,50,150)
    return cv2.addWeighted(img,0.8,edges,0.2,0)

def extract_features(img):
    h=hog(img,12,(8,8),(2,2),visualize=False,feature_vector=True)
    lbp=local_binary_pattern(img,8,1,"uniform")
    hist,_=np.histogram(lbp,bins=10,range=(0,10)); hist=hist.astype(float)
    hist/=hist.sum()+1e-7
    return np.concatenate([h,hist]).astype(np.float32)

# ---------- GUI ----------
def choose_and_run():
    root=tk.Tk(); root.withdraw()
    npz = filedialog.askopenfilename(title="Select worm_lr.npz",
                                     filetypes=[("NumPy model","*.npz")])
    if not npz: return
    W,b = load_np_model(npz)
    folder = filedialog.askdirectory(title="Select folder of PNG test images")
    if not folder: return
    imgs=[os.path.join(folder,f) for f in os.listdir(folder) if f.endswith(".png")]
    if not imgs:
        messagebox.showerror("No images","Folder contains no .png files"); return
    records=[]; cnt={0:0,1:0}
    for p in imgs:
        g=cv2.imread(p,cv2.IMREAD_GRAYSCALE); g=preprocess(g)
        f=extract_features(g).reshape(1,-1)
        lab,prob = predict_batch(f,W,b)
        lab,prob=int(lab[0]),float(prob[0])
        cnt[lab]+=1; records.append((os.path.basename(p),lab,prob))
        txt=f"{lab} ({prob*100:.1f}%)"
        cv2.putText(g,txt,(2,12),cv2.FONT_HERSHEY_SIMPLEX,0.4,255,1)
        cv2.imshow("pred",g); cv2.waitKey(1)
    cv2.destroyAllWindows()
    df = pd.DataFrame(records, columns=["filename", "label", "prob"])
    df.loc[len(df)] = ["", "", ""]
    for k in (0,1):
        df.loc[len(df)] = [f"total {k}", cnt[k], ""]

    save_path = filedialog.asksaveasfilename(
        title="Save predictions as",
        defaultextension=".xlsx",
        filetypes=[("Excel files", "*.xlsx"), ("All Files", "*.*")]
    )
    if save_path:
        df.to_excel(save_path, index=False)
        messagebox.showinfo("Done", f"Saved → {save_path}")
    else:
        messagebox.showwarning("Save Cancelled", "Excel save operation cancelled.")

if __name__=="__main__":
    choose_and_run()
