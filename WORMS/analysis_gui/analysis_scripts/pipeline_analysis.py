"""
pipeline_analysis.py
Interactive GUI to build / visualise a custom pre‑processing pipeline
for C. elegans worm detection.

Drop this file into analysis_scripts/ and run:
    python pipeline_analysis.py
or add a button in your main GUI.
"""

import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
import cv2
import numpy as np
from skimage.feature import hog
from skimage.morphology import skeletonize
from skimage.measure import regionprops, label


# ----------------- helper utilities ----------------- #
def to_uint8(img):
    img = np.clip(img, 0, 255)
    return img.astype(np.uint8)


def apply_pipeline(img_gray, opts):
    """Run selected steps in order; return list of (name, image) pairs."""
    steps = [("original", img_gray)]
    cur = img_gray.copy()

    # 1) CLAHE
    if opts["clahe"]:
        clahe = cv2.createCLAHE(
            clipLimit=opts["clahe_clip"],
            tileGridSize=(opts["clahe_tile"], opts["clahe_tile"]),
        )
        cur = clahe.apply(cur)
        steps.append(("clahe", cur.copy()))

    # 2) top‑hat
    if opts["tophat"]:
        k = cv2.getStructuringElement(
            cv2.MORPH_RECT, (opts["tophat_k"], opts["tophat_k"])
        )
        cur = cv2.morphologyEx(cur, cv2.MORPH_TOPHAT, k)
        steps.append(("tophat", cur.copy()))

    # 3) median blur
    if opts["median"]:
        cur = cv2.medianBlur(cur, opts["median_k"])
        steps.append(("median", cur.copy()))

    # 4) canny + blend
    if opts["canny"]:
        edges = cv2.Canny(cur, opts["canny_t1"], opts["canny_t2"])
        cur = cv2.addWeighted(cur, 0.8, edges, 0.2, 0)
        steps.append(("canny_blend", cur.copy()))

    # 5) threshold (Otsu OR adaptive)
    mask = None
    if opts["thresh"]:
        if opts["thresh_type"] == "otsu":
            _, mask = cv2.threshold(
                cur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
            )
        else:  # adaptive
            mask = cv2.adaptiveThreshold(
                cur,
                255,
                cv2.ADAPTIVE_THRESH_MEAN_C,
                cv2.THRESH_BINARY,
                opts["thr_block"],
                opts["thr_C"],
            )
        steps.append(("threshold", mask.copy()))
    else:
        mask = cur.copy()

    # 6) morphological open / close
    if opts["morph"]:
        se1 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, se1, iterations=opts["open_iter"])
        se2 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        mask = cv2.morphologyEx(
            mask, cv2.MORPH_CLOSE, se2, iterations=opts["close_iter"]
        )
        steps.append(("morph", mask.copy()))

    # 7) keep largest elongated contour
    if opts["contour"]:
        lbl = label(mask > 0)
        keep = np.zeros_like(mask)
        for r in regionprops(lbl):
            if r.area >= opts["min_area"] and r.eccentricity >= opts["min_ecc"]:
                keep[lbl == r.label] = 255
        mask = keep
        steps.append(("filtered_contour", mask.copy()))

    # 8) skeleton
    if opts["skeleton"]:
        skel = skeletonize(mask > 0)
        skel = (skel.astype(np.uint8) * 255)
        steps.append(("skeleton", skel.copy()))

    # 9) masked HOG visualisation
    if opts["hog"]:
        masked = cur.copy()
        masked[mask == 0] = 0
        hog_vis = hog(
            masked,
            orientations=opts["hog_ori"],
            pixels_per_cell=(opts["hog_ppc"], opts["hog_ppc"]),
            cells_per_block=(2, 2),
            visualize=True,
            feature_vector=False,
        )[1]
        hog_vis = to_uint8((hog_vis - hog_vis.min()) / (hog_vis.ptp() + 1e-6) * 255)
        steps.append(("hog", hog_vis.copy()))

    return steps


# ----------------- GUI class ----------------- #
class PipelineGUI(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Custom Worm Pre‑processing Pipeline")
        self.state("zoomed")

        # persistent image reference (avoids GC)
        self._img_ref = None

        # ------------ left controls ------------
        ctrl = tk.Frame(self, padx=10, pady=10)
        ctrl.pack(side=tk.LEFT, fill=tk.Y)

        tk.Button(ctrl, text="Load Image", command=self.load_image).pack(
            fill=tk.X, pady=5
        )
        self.img_gray = None

        self.vars = {}  # check‑box states

        # helper to add check‑box
        def add_bool(label, default=True):
            v = tk.IntVar(value=1 if default else 0)
            tk.Checkbutton(ctrl, text=label, variable=v).pack(anchor="w")
            self.vars[label] = v

        # --- steps (all default TRUE) ---
        add_bool("clahe")
        self.clahe_clip = tk.IntVar(value=2)
        self.clahe_tile = tk.IntVar(value=8)
        tk.Label(ctrl, text="  clipLimit").pack(anchor="w")
        tk.Entry(ctrl, textvariable=self.clahe_clip, width=4).pack(anchor="w")
        tk.Label(ctrl, text="  tile").pack(anchor="w")
        tk.Entry(ctrl, textvariable=self.clahe_tile, width=4).pack(anchor="w")

        add_bool("tophat")
        self.tophat_k = tk.IntVar(value=25)
        tk.Label(ctrl, text="  kernel").pack(anchor="w")
        tk.Entry(ctrl, textvariable=self.tophat_k, width=4).pack(anchor="w")

        add_bool("median")
        self.median_k = tk.IntVar(value=3)
        tk.Label(ctrl, text="  ksize").pack(anchor="w")
        tk.Entry(ctrl, textvariable=self.median_k, width=4).pack(anchor="w")

        add_bool("canny")
        self.canny_t1 = tk.IntVar(value=50)
        self.canny_t2 = tk.IntVar(value=150)
        tk.Label(ctrl, text="  t1").pack(anchor="w")
        tk.Entry(ctrl, textvariable=self.canny_t1, width=4).pack(anchor="w")
        tk.Label(ctrl, text="  t2").pack(anchor="w")
        tk.Entry(ctrl, textvariable=self.canny_t2, width=4).pack(anchor="w")

        add_bool("thresh")
        self.thr_type = tk.StringVar(value="otsu")
        tk.Radiobutton(ctrl, text="Otsu", variable=self.thr_type, value="otsu").pack(
            anchor="w"
        )
        tk.Radiobutton(
            ctrl, text="Adaptive", variable=self.thr_type, value="adaptive"
        ).pack(anchor="w")
        self.thr_block = tk.IntVar(value=21)
        self.thr_C = tk.IntVar(value=-5)
        tk.Label(ctrl, text="  block").pack(anchor="w")
        tk.Entry(ctrl, textvariable=self.thr_block, width=4).pack(anchor="w")
        tk.Label(ctrl, text="  C").pack(anchor="w")
        tk.Entry(ctrl, textvariable=self.thr_C, width=4).pack(anchor="w")

        add_bool("morph")
        self.open_iter = tk.IntVar(value=2)
        self.close_iter = tk.IntVar(value=2)
        tk.Label(ctrl, text="  open iter").pack(anchor="w")
        tk.Entry(ctrl, textvariable=self.open_iter, width=4).pack(anchor="w")
        tk.Label(ctrl, text="  close iter").pack(anchor="w")
        tk.Entry(ctrl, textvariable=self.close_iter, width=4).pack(anchor="w")

        add_bool("contour")
        self.min_area = tk.IntVar(value=200)
        self.min_ecc = tk.DoubleVar(value=0.8)
        tk.Label(ctrl, text="  min area").pack(anchor="w")
        tk.Entry(ctrl, textvariable=self.min_area, width=6).pack(anchor="w")
        tk.Label(ctrl, text="  min ecc").pack(anchor="w")
        tk.Entry(ctrl, textvariable=self.min_ecc, width=6).pack(anchor="w")

        add_bool("skeleton", False)

        add_bool("hog")
        self.hog_ori = tk.IntVar(value=12)
        self.hog_ppc = tk.IntVar(value=8)
        tk.Label(ctrl, text="  orientations").pack(anchor="w")
        tk.Entry(ctrl, textvariable=self.hog_ori, width=4).pack(anchor="w")
        tk.Label(ctrl, text="  px/cell").pack(anchor="w")
        tk.Entry(ctrl, textvariable=self.hog_ppc, width=4).pack(anchor="w")

        tk.Button(ctrl, text="Run pipeline", command=self.run_pipeline).pack(
            fill=tk.X, pady=10
        )
        tk.Button(ctrl, text="Show step‑by‑step", command=self.show_steps).pack(
            fill=tk.X
        )

        # ------------ right display (created on first show) ------------
        self.display = None
        self.steps = []  # store last run steps

    # ----------- callbacks ----------- #
    def load_image(self):
        path = filedialog.askopenfilename(
            filetypes=[("Image", "*.png;*.jpg;*.jpeg")]
        )
        if path:
            img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
            if img is None:
                messagebox.showerror("Error", "Cannot load image")
                return
            self.img_gray = img
            self.show_image(img)

    def collect_opts(self):
        return dict(
            clahe=self.vars["clahe"].get(),
            clahe_clip=self.clahe_clip.get(),
            clahe_tile=self.clahe_tile.get(),
            tophat=self.vars["tophat"].get(),
            tophat_k=self.tophat_k.get(),
            median=self.vars["median"].get(),
            median_k=self.median_k.get(),
            canny=self.vars["canny"].get(),
            canny_t1=self.canny_t1.get(),
            canny_t2=self.canny_t2.get(),
            thresh=self.vars["thresh"].get(),
            thresh_type=self.thr_type.get(),
            thr_block=self.thr_block.get(),
            thr_C=self.thr_C.get(),
            morph=self.vars["morph"].get(),
            open_iter=self.open_iter.get(),
            close_iter=self.close_iter.get(),
            contour=self.vars["contour"].get(),
            min_area=self.min_area.get(),
            min_ecc=self.min_ecc.get(),
            skeleton=self.vars["skeleton"].get(),
            hog=self.vars["hog"].get(),
            hog_ori=self.hog_ori.get(),
            hog_ppc=self.hog_ppc.get(),
        )

    def run_pipeline(self):
        if self.img_gray is None:
            messagebox.showinfo("No image", "Load an image first.")
            return
        self.steps = apply_pipeline(self.img_gray, self.collect_opts())
        combo = np.hstack([self.steps[0][1], self.steps[-1][1]])
        self.show_image(combo)

    def show_steps(self):
        if not self.steps:
            messagebox.showinfo("Run first", "Run the pipeline first.")
            return
        win = tk.Toplevel(self)
        win.title("Pipeline stages")
        cols = 2  # number of columns for steps display
        for idx, (name, img) in enumerate(self.steps):
            imgtk = ImageTk.PhotoImage(Image.fromarray(to_uint8(img)))
            frame = tk.Frame(win, bd=2, relief=tk.SUNKEN, padx=5, pady=5)
            frame.grid(row=idx // cols, column=idx % cols, padx=5, pady=5, sticky="nsew")
            tk.Label(frame, text=name, font=("Arial", 12, "bold")).pack()
            lbl = tk.Label(frame, image=imgtk)
            lbl.image = imgtk  # keep reference
            lbl.pack()

    # ------------ util ------------ #
    def show_image(self, img_np):
        imgtk = ImageTk.PhotoImage(Image.fromarray(to_uint8(img_np)))
        self._img_ref = imgtk  # prevent GC
        if self.display is None:
            self.display = tk.Label(self, image=imgtk)
            self.display.pack(side=tk.RIGHT, expand=True)
        else:
            self.display.configure(image=imgtk)


# ----------------- run ----------------- #
if __name__ == "__main__":
    PipelineGUI().mainloop()
