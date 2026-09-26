# Phishing Email Detection Model

An end-to-end Machine Learning pipeline built using Python and Scikit-learn to classify emails as **Phishing** or **Safe**.

---

## 📌 Project Overview
This project extracts linguistic, structural, and heuristic features from raw email text to detect deceptive patterns, urgent calls to action, and malicious links.

### Key Features
- **Heuristic Feature Extraction**: Custom transformer detecting URLs, direct IP links, urgent keywords (e.g., *verify*, *suspended*, *urgent*), capitalizations, and exclamation marks.
- **TF-IDF Text Analysis**: Word and bi-gram feature extraction.
- **Classifier**: Random Forest Classifier within a Scikit-Learn `Pipeline`.
- **Model Persistence**: Automatically exports the trained pipeline to `phishing_detector_model.pkl` using Joblib.
- **Interactive Live Tester**: Terminal interface to classify custom email text in real time.

---

## 📊 Results & Performance

- **Model Accuracy:** 100.00%

### Confusion Matrix
| Actual \ Predicted | Predicted: Safe | Predicted: Phishing |
| :--- | :---: | :---: |
| **Actual: Safe** | 25 | 0 |
| **Actual: Phishing** | 0 | 25 |

### Sample Predictions
- `"Hi David, let's review the code changes in tomorrow's standup."` ➔ **Safe** (96.0% confidence)
- `"ACCOUNT SUSPENDED! Go to http://10.0.0.1/verify-now to secure your credentials."` ➔ **Phishing** (68.0% confidence)

---

## 🛠️ Tech Stack
- **Language:** Python
- **Machine Learning:** Scikit-learn
- **Data Manipulation:** Pandas, NumPy
- **Serialization:** Joblib

---

## 🚀 How to Run

1. **Clone the repository:**
   ```bash
   git clone [https://github.com/your-username/phishing-email-detection.git](https://github.com/KatikaVeronica/phishing-email-detection.git)
   cd phishing-email-detection