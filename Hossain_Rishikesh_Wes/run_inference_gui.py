# run_inference_gui.py
import tkinter as tk
from tkinter import messagebox
from inference_scripts.elegans_inference import run_elegans_gui
# from inference_scripts.mnist_inference import run_mnist_gui  # <‑‑ when ready

class MainGUI(tk.Tk):                 
    def __init__(self):
        super().__init__()
        self.title("Inference Launcher")
        self.geometry("400x250")
        tk.Label(self, text="Select dataset to test:", font=("Arial", 16)).pack(pady=25)

        tk.Button(self, text="Test  C. elegans",
                  font=("Arial", 14), width=20,
                  command=self.run_elegans).pack(pady=10)

        tk.Button(self, text="Test  MNIST (coming soon)",
                  font=("Arial", 14), width=20,
                  state=tk.DISABLED  # enable when mnist script ready
                  # command=self.run_mnist
                  ).pack(pady=10)

        tk.Button(self, text="Quit", font=("Arial", 12),
                  command=self.destroy).pack(pady=20)

    def run_elegans(self):
        self.withdraw()
        run_elegans_gui()
        self.deiconify()

    # def run_mnist(self):
    #     self.withdraw()
    #     run_mnist_gui()
    #     self.deiconify()

if __name__ == "__main__":
    MainGUI().mainloop()
