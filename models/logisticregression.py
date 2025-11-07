
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

# fake data
data = pd.DataFrame({
    "pupil_dilation": [0.30, 0.45, 0.25, 0.50, 0.33, 0.48, 0.29, 0.60, 0.31, 0.52],
    "reaction_time": [1.0, 1.6, 0.8, 1.9, 1.2, 1.7, 0.9, 2.0, 1.1, 1.8],
    "fixation_stability": [0.85, 0.40, 0.90, 0.35, 0.80, 0.45, 0.95, 0.30, 0.82, 0.38],
    "label": ["low", "high", "low", "high", "low", "high", "low", "high", "low", "high"]
})

# split features and labels
X = data[["pupil_dilation", "reaction_time", "fixation_stability"]]
y = data["label"]

# train test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42
)

# training the model
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# checking the model
y_pred = model.predict(X_test)
acc = accuracy_score(y_test, y_pred)
print("Validation Accuracy:", acc)
print("Predictions:", y_pred)

# confusion matrix
cm = confusion_matrix(y_test, y_pred, labels=model.classes_)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=model.classes_, yticklabels=model.classes_)
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix - Logistic Regression (mock)")
plt.show()

# print out the repot
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# save the model
data.to_csv("fake_eye_data_logistic.csv", index=False)
joblib.dump(model, "logistic_regression_mock.pkl")
print("Saved model and fake data.")
