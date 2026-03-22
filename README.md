# Custom NLP Bank Transaction Classifier

A complete, end-to-end Machine Learning pipeline built from scratch in Python. This project implements a custom Term Frequency-Inverse Document Frequency (TF-IDF) Vectorizer and a Multinomial Naive Bayes Classifier using pure linear algebra (`NumPy`) to categorize bank transactions based on their raw text descriptions.

## 🚀 Key Features

* **From-Scratch NLP Pipeline:** Bypasses black-box libraries to manually engineer text cleaning, matrix vectorization, and probabilistic classification.
* **Data Leakage Prevention:** Strictly enforces the `fit_transform` and `transform` architecture. The vectorizer locks its vocabulary during training to ensure out-of-vocabulary (OOV) words in the test set do not break matrix dimensions.
* **Vectorized Math:** Replaces slow, iterative `for` loops with optimized `NumPy` matrix operations (Dot Products, Column Stacking) for lightning-fast training and prediction phases.
* **Laplace Smoothing:** Implements additive smoothing (+1) natively within the Naive Bayes likelihood calculations to handle zero-probability edge cases.
* **Hardware Safe:** Utilizes logarithmic addition to calculate final category scores, preventing the floating-point underflow crashes common in raw probability multiplication.

## 🧠 Architecture

The project is divided into two primary custom engines:

### 1. `TFIDFAnalyzer`
A custom text vectorizer that converts messy bank transaction strings into mathematical matrices.
* **Cleaning:** Uses regex sweeps to remove special characters, numbers, and convert text to lowercase.
* **Vocabulary:** Builds a unique, alphabetized dictionary mapping based *only* on the training data.
* **TF-IDF Matrix:** Calculates the proportional frequency of words in a document (TF) and multiplies it by the logarithmic rarity of the word across the entire dataset (IDF).

### 2. `NaiveBayesClassifer`
A custom probabilistic machine learning model.
* **Fit (Training):** Calculates the Log-Priors (baseline category probabilities) and Log-Likelihoods (probability of each word given a category) using boolean masking.
* **Predict (Testing):** Computes the dot product of the incoming TF-IDF matrix against the trained log-likelihoods, adds the priors, and uses `numpy.argmax` across the columns to declare the winning category.

## 🛠️ Requirements

* `Python 3.x`
* `numpy` (For core mathematical and matrix operations)
* `pandas` (For initial CSV data loading)
* `scikit-learn` (Used *only* for the `train_test_split` utility and `accuracy_score` metric)

## 📂 Usage

1. Ensure you have your dataset saved as `bank_transactions.csv` in the root directory. The CSV should contain at least two columns: `Company` (the raw text) and `Category` (the target label).
2. Run the main execution pipeline:

```bash
python main.py

Pipeline Execution Flow:
Loads the CSV and splits data into 80% Training and 20% Testing subsets.

Initializes the custom TFIDFAnalyzer and NaiveBayesClassifer.

Calls fit_transform() to build the locked vocabulary and format the training matrix.

Trains the Naive Bayes model on the training matrix.

Calls transform() to safely format the test data, ignoring unseen words to protect matrix geometry.

Generates predictions and outputs the final accuracy score.

👨‍💻 Author
Jash Kotadiya ```
