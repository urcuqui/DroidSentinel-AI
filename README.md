# DroidSentinel-AI

DroidSentinel-AI is a Streamlit app that trains a Random Forest classifier for Android malware detection using permission data. It lets you explore the 20 most important permissions and simulate new apps by toggling the permissions they request.

## Features
- Offline training on your `train.csv` dataset (RandomForest, cached for faster reloads).
- Shows the top 20 most important permissions according to the trained model.
- Simple sidebar UI to toggle requested permissions; all others default to 0 if you choose.
- Predicts benign (0) vs malware (1) and reports class probabilities.

## Requirements
- Python 3.9+
- Dependencies: `streamlit`, `pandas`, `numpy`, `scikit-learn` (see `requirements.txt`).

## Setup
1) Clone this repo and open a terminal in its folder.
2) (Optional) Create and activate a virtual environment.
3) Install dependencies: `pip install -r requirements.txt`.

## Dataset format (`train.csv`)
- Location: repository root.
- Layout: a single CSV column whose header lists all permission feature names plus `type`, separated by semicolons (`;`).
- Each row: semicolon-separated numeric values matching the header order. `type` is the label: `0` = benign, `1` = malware.
- Minimal example:
  - Header: `perm_A;perm_B;perm_C;type`
  - Row: `1;0;1;0`

## Run the app
- Launch: `streamlit run main.py`
- Streamlit will open in your browser (port 8501 by default).

## Usage
- Use the sidebar to select the permissions your APK requests (Top 20 shown for convenience).
- Leave "Set all non-selected permissions to 0" enabled to assume the rest are absent.
- Click "Predict app type" to get the predicted label and probabilities.

## Notes and limitations
- Model quality depends entirely on the quality and balance of `train.csv`; retrain with recent, representative samples.
- This tool checks declared permissions only; it does not replace dynamic analysis or a full security review of the APK.
