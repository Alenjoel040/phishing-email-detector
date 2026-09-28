import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report


# 1. Load the dataset
data = pd.read_csv("data/phishing_emails.csv")

# 2. Remove empty rows
data = data.dropna(subset=["email", "label"])

# 3. Separate email text and labels
X = data["email"]
y = data["label"]

# 4. Convert labels into numbers
y = y.map({
    "legitimate": 0,
    "phishing": 1
})

# 5. Split the dataset into training and testing data
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# 6. Create the machine-learning pipeline
model = Pipeline([
    (
        "tfidf",
        TfidfVectorizer(
            lowercase=True,
            stop_words="english"
        )
    ),
    (
        "classifier",
        LogisticRegression(max_iter=1000)
    )
])

# 7. Train the model
model.fit(X_train, y_train)

# 8. Test the model
y_pred = model.predict(X_test)

# 9. Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)

print("===================================")
print(" PHISHING EMAIL DETECTION MODEL")
print("===================================")

print("\nAccuracy:", accuracy)

print("\nClassification Report:")
print(
    classification_report(
        y_test,
        y_pred,
        target_names=["Legitimate", "Phishing"],
        zero_division=0
    )
)

# 10. Save the trained model
joblib.dump(model, "phishing_model.pkl")

print("\nModel saved successfully!")
print("File created: phishing_model.pkl")