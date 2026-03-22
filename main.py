import pandas
import numpy
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from TFIDFAnalyzer import TFIDFAnalyzer
from NaiveBayesClassifer import NaiveBayesClassifer

# If your classes are in a different file called models.py, uncomment the line below:
# from models import TFIDFAnalyzer, NaiveBayesClassifer

def main():
    print("1. Loading the Bank Transactions dataset...")
    try:
        dataFrame = pandas.read_csv("bank_transactions.csv")
    except FileNotFoundError:
        print("Error: Could not find 'bank_transactions.csv'. Make sure it is in the same folder!")
        return

    # Extract the raw text strings and the correct category answers
    X_raw = list(dataFrame["Company"]) 
    Y = numpy.array(dataFrame["Category"])

    print("2. Slicing the data (80% Training, 20% Testing)...")
    # random_state=42 ensures the shuffle is the exact same every time you run it, making debugging easier
    X_train_raw, X_test_raw, Y_train, Y_test = train_test_split(X_raw, Y, test_size=0.2, random_state=42)

    print("3. Powering up custom NLP engines...")
    vectorizer = TFIDFAnalyzer()
    model = NaiveBayesClassifer()


    print("4. Vectorizer: Reading training data and building the locked dictionary...")
    X_train_matrix = vectorizer.fitTransform(X_train_raw) 

    print("5. Naive Bayes: Calculating Log-Priors and Likelihoods...")
    model.fit(X_train_matrix, Y_train)


    print("6. Vectorizer: Formatting test data using the locked dictionary...")
    X_test_matrix = vectorizer.transform(X_test_raw)

    print("7. Naive Bayes: Predicting categories for unseen transactions...")
    predictions = model.predict(X_test_matrix)

 
    final_score = accuracy_score(Y_test, predictions)

    print("\n==========================================")
    print(f"FINAL MODEL ACCURACY: {round(final_score * 100, 2)}% ")
    print("==========================================")

    print("\nSample Output (First 5 Test Transactions):")
    for i in range(5):
        print(f"Transaction: '{X_test_raw[i]}'")
        print(f"  -> Model Guessed: {predictions[i]}")
        print(f"  -> Actual Answer: {Y_test[i]}\n")

if __name__ == "__main__":
    main()