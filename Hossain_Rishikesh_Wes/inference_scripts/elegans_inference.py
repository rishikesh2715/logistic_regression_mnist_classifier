# inference_scripts/elegans_inference.py
import os, sys, cv2, numpy as np, pandas as pd, tkinter as tk
from tkinter import filedialog, messagebox
from joblib import load
from skimage.feature import hog, local_binary_pattern

# ---------------- preprocessing & features ---------------- #
def preprocess(img):
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    img = clahe.apply(img)
    img = cv2.medianBlur(img, 3)
    edges = cv2.Canny(img, 50, 150)
    return cv2.addWeighted(img, 0.8, edges, 0.2, 0)

def extract_features(img):
    hog_feats = hog(img, orientations=12,
                    pixels_per_cell=(8,8),
                    cells_per_block=(2,2),
                    visualize=False, feature_vector=True)
    lbp = local_binary_pattern(img, P=8, R=1, method="uniform")
    lbp_hist,_ = np.histogram(lbp, bins=10, range=(0,10))
    lbp_hist = lbp_hist.astype(float); lbp_hist /= (lbp_hist.sum()+1e-7)
    return np.concatenate([hog_feats, lbp_hist]).astype(np.float32)

# ---------------- helper to predict one image -------------- #
def predict(model, img_path):
    g = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    if g is None:
        raise ValueError(f"Cannot read {img_path}")
    g = preprocess(g)
    feat = extract_features(g).reshape(1,-1)
    pred = model.predict(feat)[0]
    proba = model.predict_proba(feat)[0][int(pred)] if hasattr(model,"predict_proba") else None
    return int(pred), proba

def overlay_and_show(img_path, pred, proba):
    img = cv2.imread(img_path)
    if img is None: return
    txt = f"Pred: {pred}"
    if proba is not None: txt += f" ({proba*100:.1f}%)"
    cv2.putText(img, txt, (10,30), cv2.FONT_HERSHEY_SIMPLEX, 1,
                (0,255,0), 2)
    cv2.imshow("Prediction", img); cv2.destroyAllWindows()

# ---------------- GUI wrapper ---------------- #
def run_elegans_gui():
    root = tk.Tk(); root.withdraw()

    # prompt model
    model_path = filedialog.askopenfilename(
        title="Select trained .joblib model",
        filetypes=[("Joblib","*.joblib"),("All","*.*")]
    )
    if not model_path:
        messagebox.showinfo("Info","Model not selected."); return
    try:
        model = load(model_path)
    except Exception as e:
        messagebox.showerror("Error", f"Could not load model:\n{e}"); return

    # prompt image directory
    img_dir = filedialog.askdirectory(title="Select directory of PNG test images")
    if not img_dir:
        messagebox.showinfo("Info","Directory not selected."); return
    img_files = [os.path.join(img_dir,f) for f in os.listdir(img_dir)
                 if f.lower().endswith(".png")]
    if not img_files:
        messagebox.showerror("Error","No .png files found."); return

    # run inference
    results = []
    counts = {0:0, 1:0}
    for p in img_files:
        try:
            pred, proba = predict(model, p)
            results.append((os.path.basename(p), pred))
            counts[pred] += 1
            overlay_and_show(p, pred, proba)
        except Exception as e:
            print("Error on", p, ":", e)

    # save to Excel
    df = pd.DataFrame(results, columns=["filename","label"])
    df.loc[len(df)] = ["", ""]  # blank row
    for lbl in sorted(counts):
        df.loc[len(df)] = [f"Total label {lbl}", counts[lbl]]

    excel_path = os.path.join(
        os.path.dirname(model_path),
        f"elegans_predictions_{len(results)}.xlsx"
    )
    df.to_excel(excel_path, index=False)
    messagebox.showinfo("Done", f"Excel saved:\n{excel_path}")
    print("Excel saved to:", excel_path)

# allow CLI run too
if __name__ == "__main__":
    run_elegans_gui()
