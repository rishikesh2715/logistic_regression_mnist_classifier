# run_inference_gui.py
from tkinter import messagebox
from inference_scripts.elegans_inference import run_elegans_gui
import tkinter as tk
from tkinter import filedialog, messagebox
import threading
from inference_scripts.mnist_inference import run_mnist_inference

class MainGUI(tk.Tk):                 
    def __init__(self):
        super().__init__()
        self.title("Inference Launcher")
        self.geometry("400x250")
        tk.Label(self, text="Select dataset to test:", font=("Arial", 16)).pack(pady=25)

        tk.Button(self, text="Test  C. elegans",
                  font=("Arial", 14), width=20,
                  command=self.run_elegans).pack(pady=10)

        tk.Button(self, text="Test  MNIST",
          font=("Arial", 14), width=20,
          command=self.run_mnist).pack(pady=10)


        tk.Button(self, text="Quit", font=("Arial", 12),
                  command=self.destroy).pack(pady=20)

    def run_elegans(self):
        self.withdraw()
        run_elegans_gui()
        self.deiconify()

    def run_mnist(self):
        image_folder = filedialog.askdirectory(title="Select MNIST Test Image Folder")
        if not image_folder:
            messagebox.showwarning("No Folder Selected", "Please select a folder containing MNIST images.")
            return
        
        model_file = filedialog.askopenfilename(title="Select Trained Model", filetypes=[("Pickle Files", "*.pkl")])
        if not model_file:
            messagebox.showwarning("No Model Selected", "Please select a trained model file.")
            return
        
        self.withdraw()  
        run_mnist_inference(image_folder, model_file) 
        self.deiconify()  

if __name__ == "__main__":
    MainGUI().mainloop()
