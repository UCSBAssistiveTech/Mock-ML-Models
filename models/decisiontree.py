
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import joblib

# 1. create data
data = pd.DataFrame({
    "pupil_dilation": [0.30, 0.45, 0.25, 0.50, 0.33, 0.48, 0.29, 0.60, 0.31, 0.52],
    "reaction_time": [1.0, 1.6, 0.8, 1.9, 1.2, 1.7, 0.9, 2.0, 1.1, 1.8],
    "fixation_stability": [0.85, 0.40, 0.90, 0.35, 0.80, 0.45, 0.95, 0.30, 0.82, 0.38],
    "label": ["low", "high", "low", "high", "low", "high", "low", "high", "low", "high"]
})

# 2. Split features and labels
X = data[["pupil_dilation", "reaction_time", "fixation_stability"]]
y = data["label"]

# 3. Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42
)

# 4. Train model
model = DecisionTreeClassifier(max_depth=3, random_state=42)
model.fit(X_train, y_train)

# 5. Evaluate model
y_pred = model.predict(X_test)
acc = accuracy_score(y_test, y_pred)
print("Validation Accuracy:", acc)
print("Predictions:", y_pred)

# 6. Visualize decision tree
plt.figure(figsize=(10, 6))
plot_tree(
    model,
    feature_names=X.columns,
    class_names=model.classes_,
    filled=True,
    rounded=True
)
plt.title("Decision Tree for Alzheimer’s Detection (mock)")
plt.show()

# 7. Save model and data
data.to_csv("fake_eye_data.csv", index=False)
joblib.dump(model, "decision_tree_mock.pkl")
print("Saved model and fake data.")
