import tkinter as tk
from analysis_scripts.hog_analysis import HOGAnalysisGUI
from analysis_scripts.lbp_analysis import LBPAnalysisGUI
from analysis_scripts.histogram_analysis import HistogramAnalysisGUI
from analysis_scripts.binarization_analysis import BinarizationAnalysisGUI
from analysis_scripts.frequency_analysis import FrequencyAnalysisGUI

def open_hog_analysis(main_window):
    # Hide main window and open HOG analysis window
    main_window.withdraw()
    HOGAnalysisGUI(main_window)

def open_lbp_analysis(main_window):
    # Hide main window and open LBP analysis window
    main_window.withdraw()
    LBPAnalysisGUI(main_window)

def open_histogram_analysis(main_window):
    # Hide main window and open Histogram analysis window
    main_window.withdraw()
    HistogramAnalysisGUI(main_window)

def open_binarization_analysis(main_window):
    # Hide main window and open Binarization analysis window
    main_window.withdraw()
    BinarizationAnalysisGUI(main_window)

def open_frequency_analysis(main_window):
    # Hide main window and open Frequency analysis window
    main_window.withdraw()
    FrequencyAnalysisGUI(main_window)

if __name__ == '__main__':
    root = tk.Tk()
    root.title("Analysis GUI Main")
    root.geometry("800x600")
    
    header = tk.Label(root, text="Select an Analysis Module", font=("Arial", 18, "bold"))
    header.pack(pady=20)
    
    # Button for HOG Analysis
    btn_hog = tk.Button(
        root, 
        text="HOG Analysis", 
        command=lambda: open_hog_analysis(root), 
        font=("Arial", 16), 
        width=20
    )
    btn_hog.pack(pady=10)
    
    # Button for LBP Analysis
    btn_lbp = tk.Button(
        root, 
        text="LBP Analysis", 
        command=lambda: open_lbp_analysis(root), 
        font=("Arial", 16), 
        width=20
    )
    btn_lbp.pack(pady=10)

    # Button for Histogram Analysis
    btn_histogram = tk.Button(
        root, 
        text="Histogram Analysis", 
        command=lambda: open_histogram_analysis(root), 
        font=("Arial", 16), 
        width=20
    )
    btn_histogram.pack(pady=10)

    # Button for Binarization Analysis
    btn_binarization = tk.Button(
        root, 
        text="Binarization Analysis", 
        command=lambda: open_binarization_analysis(root), 
        font=("Arial", 16), 
        width=20
    )
    btn_binarization.pack(pady=10)

    # Button for Frequency Analysis
    btn_frequency = tk.Button(
        root, 
        text="Frequency Analysis", 
        command=lambda: open_frequency_analysis(root), 
        font=("Arial", 16), 
        width=20
    )
    btn_frequency.pack(pady=10)
    
    root.mainloop()
