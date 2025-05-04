
# 🎧 Grammar Scoring Engine (Audio-Based)

This project is a **Grammar Quality Scoring Engine** that predicts the quality of spoken English responses using audio files. It uses **MFCC audio features**, handles class imbalance with **SMOTE**, trains a **Random Forest classifier**.

---

## 📁 Project Structure

```
data/
├── train.csv
├── test.csv
├── audios/
│   ├── train/
│   └── test/
models/
├── best_model.pkl
├── scaler.pkl
notebooks/
├── GrammarScoring.ipynb
README.md
```

---

## 🚀 Features

- ✅ MFCC feature extraction using `librosa`
- ✅ Class balancing using `SMOTE`
- ✅ Classification with `RandomForestClassifier`
- ✅ Hyperparameter tuning via `GridSearchCV`
- ✅ Web interface using Flask to upload and predict on new audio
- ✅ Model & Scaler persistence using `joblib`

---

## 🧪 Model Training

```python
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.preprocessing import StandardScaler
```

1. **Preprocess data** and extract MFCCs  
2. **Train/Test split** with stratification  
3. **Normalize** using `StandardScaler`  
4. **Apply SMOTE** for class balancing  
5. **Train & Tune model** using `GridSearchCV`

---

## 🌐 Run the Web App

1. Install dependencies:
   ```bash
   pip install flask librosa scikit-learn imbalanced-learn
   ```

2. Run Flask app:
   ```bash
   python app/app.py
   ```

3. Visit `http://127.0.0.1:5000` to use the interface.

---

## 📝 Requirements

- Python 3.8+
- `librosa`
- `scikit-learn`
- `imbalanced-learn`
- `flask`
- `joblib`
- `pandas`
- `numpy`

You can install all dependencies with:
```bash
pip install -r requirements.txt
```

---

## 📊 Sample Prediction Output

Once you upload an audio file, the app returns:
- Predicted grammar quality label (`Good`, `Moderate`, `Poor`, etc.)
- Debug info (feature vector shape, file path)

---

## 🙌 Acknowledgements

- Dataset: [Kaggle - Spoken English Grammar Quality](https://www.kaggle.com/)
- Libraries: `librosa`, `scikit-learn`, `Flask`, `imbalanced-learn`

---

## 🔖 License

MIT License. Feel free to use, modify, and share.
