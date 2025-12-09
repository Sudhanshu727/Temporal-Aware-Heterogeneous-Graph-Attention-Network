Financial Fraud Detection – TA-HGAT Implementation

This folder contains the implementation for the Developer Round 1 assignment.

📁 Files

fraud_detection_model.py
Main Python script containing:

Data Generator

Unique Model (TA-HGAT)

Training Logic

requirements.txt
List of all required Python libraries.

🚀 How to Run

1. Install dependencies
   pip install -r requirements.txt

Note:
For torch_geometric, if you face installation errors, refer to the official PyG website for OS-specific installation commands.

2. Run the model
   python fraud_detection_model.py

📊 Output

Running the script will:

1. Print training logs & final metrics

Metrics displayed in console:

AUC

F1 Score

Recall

2. Generate two visualizations:

confusion_matrix_ta-hgat_proposed_model.png
Shows the model’s classification performance.

comparative_analysis.png
Bar chart comparing the TA-HGAT model with baseline models.

📝 Using This for the Assignment
Screenshots

Take screenshots of:

Console output showing the high AUC/Recall.

The two generated image files.

Paste these into the “Visualizations” section of your main document.

Code Submission

Zip:

fraud_detection_model.py

Your Word document

Then submit the zipped folder as your final submission.
