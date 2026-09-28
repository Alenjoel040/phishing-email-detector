import joblib

# Load the trained model
model = joblib.load("phishing_model.pkl")

print("===================================")
print(" PHISHING EMAIL DETECTOR")
print("===================================")

# Ask the user to enter an email
email = input("\nEnter the email text:\n")

# Make prediction
prediction = model.predict([email])[0]

# Display result
if prediction == 1:
    print("\nResult: PHISHING EMAIL")
    print("Warning: This email may be dangerous.")
else:
    print("\nResult: LEGITIMATE EMAIL")
    print("This email appears to be legitimate.")