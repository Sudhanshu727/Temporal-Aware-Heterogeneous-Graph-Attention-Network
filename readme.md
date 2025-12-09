# Financial Fraud Detection – TA-HGAT Implementation

This folder contains the implementation for the **Developer Round 1** assignment.

---

## Files

- **fraud_detection_model.py**  
  The main Python script containing:
  - Data Generator  
  - Unique Model (TA-HGAT)  
  - Training Logic  

- **requirements.txt**  
  List of Python libraries needed.

---

## How to Run

### 1. Install the dependencies:
```bash
pip install -r requirements.txt
```
Note: For torch_geometric, if you encounter errors, please visit the PyG website for specific installation commands matching your OS.

### 2. Run the model:
```bash
python fraud_detection_model.py
```
## Output
The script will print training logs and final metrics (AUC, F1, Recall) to the console.

It will also generate two image files in the same directory:

confusion_matrix_ta-hgat_proposed_model.png
Visual evidence of the model's classification performance.

comparative_analysis.png
A bar chart comparing your model against baselines.
