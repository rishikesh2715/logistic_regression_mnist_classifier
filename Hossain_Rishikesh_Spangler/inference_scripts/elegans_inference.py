import os, cv2, numpy as np, pandas as pd, tkinter as tk
from tkinter import filedialog, messagebox
from skimage.feature import hog, local_binary_pattern

# optional back‑ends
try:                       from joblib import load as jl_load
except ImportError:        jl_load = None
try:                       import onnxruntime as ort
except ImportError:        ort = None

# ──────────────────────────────────────────────────────────────
# 1.  Pre‑processing & features  (unchanged)
# ──────────────────────────────────────────────────────────────
def preprocess(img):
    clahe = cv2.createCLAHE(2.0, (8, 8)); img = clahe.apply(img)
    img   = cv2.medianBlur(img, 3)
    edges = cv2.Canny(img, 50, 150)
    return cv2.addWeighted(img, 0.8, edges, 0.2, 0)

def extract_features(img):
    hog_feat = hog(img, 12, (8, 8), (2, 2), visualize=False, feature_vector=True)
    lbp      = local_binary_pattern(img, 8, 1, "uniform")
    hist, _  = np.histogram(lbp, bins=10, range=(0, 10))
    hist     = hist.astype(float); hist /= hist.sum() + 1e-7
    return np.concatenate([hog_feat, hist]).astype(np.float32)

# ──────────────────────────────────────────────────────────────
# 2.  Model loader that handles  .npz  •  .onnx  •  .joblib
# ──────────────────────────────────────────────────────────────
class WormModel:
    def __init__(self, path):
        self.path = path
        self.ext  = os.path.splitext(path)[1].lower()

        if self.ext == ".npz":                         # ─── NumPy weights ───
            d = np.load(path)
            self.W = d["W"]; self.b = float(d["b"])
            self.backend = "np"

        elif self.ext == ".onnx":                      # ─── ONNX Runtime ────
            if ort is None:
                raise RuntimeError("onnxruntime not installed")
            sess  = ort.InferenceSession(path, providers=["CPUExecutionProvider"])
            self.sess, self.in_name, self.out_name = sess, sess.get_inputs()[0].name, sess.get_outputs()[0].name
            self.backend = "onnx"

        else:                                          # ─── joblib / pickle ─
            if jl_load is None:
                raise RuntimeError("joblib not installed")
            self.clf = jl_load(path)
            self.backend = "sk"

    # ---------------------------------------------------------
    def predict(self, feat_vec):
        """feat_vec shape (1, n_features)  →  (label:int, prob:float|None)"""
        if self.backend == "np":
            p = 1 / (1 + np.exp(-(feat_vec @ self.W + self.b)))[0]
            return int(p >= 0.5), float(p)

        elif self.backend == "onnx":
            out = self.sess.run([self.out_name],
                                {self.in_name: feat_vec.astype(np.float32)})[0]
            out = np.asarray(out)
            if out.ndim == 1: out = out[np.newaxis, :]
            lab = int(out.argmax(1)[0]); prob = float(out[0, lab])
            return lab, prob

        else:  # scikit‑learn
            lab  = int(self.clf.predict(feat_vec)[0])
            prob = float(self.clf.predict_proba(feat_vec)[0][lab]) \
                   if hasattr(self.clf, "predict_proba") else None
            return lab, prob

# ──────────────────────────────────────────────────────────────
# 3.  Single‑image helper
# ──────────────────────────────────────────────────────────────
def predict_image(model, img_path):
    g = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    if g is None: raise ValueError(f"Cannot read {img_path}")
    g = preprocess(g)
    feat = extract_features(g).reshape(1, -1)
    return model.predict(feat)

# ──────────────────────────────────────────────────────────────
# 4.  GUI
# ──────────────────────────────────────────────────────────────
def overlay_and_show(img_path, pred, prob):
    img = cv2.imread(img_path);  txt = f"Pred: {pred}"
    if prob is not None: txt += f" ({prob*100:.1f}%)"
    cv2.putText(img, txt, (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2)
    cv2.imshow("Prediction", img); cv2.waitKey(1)

class InferenceGUI(tk.Tk):
    def __init__(self, model):
        super().__init__()
        self.model = model
        self.title("C. elegans inference")
        self.geometry("400x220")
        tk.Label(self, text="Select inference method:", font=("Arial",14)).pack(pady=20)
        tk.Button(self, text="Individual images", font=("Arial",12),
                  command=self.pick_images).pack(pady=5, fill=tk.X, padx=60)
        tk.Button(self, text="Directory of images", font=("Arial",12),
                  command=self.pick_dir).pack(pady=5, fill=tk.X, padx=60)
        tk.Button(self, text="Quit", font=("Arial",12),
                  command=self.destroy).pack(pady=15)

    def pick_images(self):
        files = filedialog.askopenfilenames(title="Select PNG images",
                                            filetypes=[("PNG","*.png")])
        if files: self.run(files)

    def pick_dir(self):
        d = filedialog.askdirectory(title="Select folder of PNG images")
        if not d: return
        files = [os.path.join(d,f) for f in os.listdir(d) if f.lower().endswith(".png")]
        if files: self.run(files)
        else: messagebox.showerror("No PNG","No .png images found in that folder.")

    def run(self, paths):
        records, counts = [], {0:0, 1:0}
        for p in paths:
            try:
                lab, prob = predict_image(self.model, p)
                records.append((os.path.basename(p), lab))
                counts[lab] += 1
                overlay_and_show(p, lab, prob)
            except Exception as e:
                print("Error on", p, ":", e)

        # ─── Excel output  (save into <project‑root>/outputs) ───────────────
        df = pd.DataFrame(records, columns=["filename", "label"])
        df.loc[len(df)] = ["", ""]                       # blank separator
        for k in (0, 1):
            df.loc[len(df)] = [f"total {k}", counts[k]]

        # project‑root = one level above  inference_scripts/
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        out_folder   = os.path.join(project_root, "outputs")
        os.makedirs(out_folder, exist_ok=True)

        out_path = os.path.join(out_folder, "prediction_elegans.xlsx")
        df.to_excel(out_path, index=False)

        messagebox.showinfo("Done",
                            f"Excel saved to:\n{out_path}")
        print("Excel saved to:", out_path)
        self.destroy()

# ──────────────────────────────────────────────────────────────
def run_gui():
    root = tk.Tk(); root.withdraw()
    mdl = filedialog.askopenfilename(
        title="Select model (.npz  / .onnx  / .joblib)",
        filetypes=[("Model files","*.npz *.onnx *.joblib"), ("All","*.*")]
    )
    if not mdl: return
    try:
        model = WormModel(mdl)
    except Exception as e:
        messagebox.showerror("Error", f"Could not load model:\n{e}")
        return
    InferenceGUI(model).mainloop()

if __name__ == "__main__":
    run_gui()
