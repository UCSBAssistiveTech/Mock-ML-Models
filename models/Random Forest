"""
rf_eye_predictor.py

Random Forest example for Alzheimer’s prediction from eye-tracking features.

Expected CSV format (example columns):
 subject_id, age, sex, fixation_duration_mean, fixation_count,
 saccade_length_mean, pupil_diameter_mean, scanpath_length, label

label: 0 = healthy, 1 = Alzheimer's (or adapt accordingly)
"""

import os
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
    classification_report, confusion_matrix
)
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
import joblib
import warnings

warnings.filterwarnings("ignore")  # remove in production

# Optional: uncomment if you want SMOTE balancing
# from imblearn.over_sampling import SMOTE

RANDOM_STATE = 42
DATA_PATH = "eye_data.csv"  # change to your path
MODEL_OUT = "rf_eye_model.joblib"


def generate_synthetic_eye_data(n_subjects=800, imbalance_ratio=0.2, seed=RANDOM_STATE):
    """
    Creates a synthetic dataset with plausible eye-tracking features.
    imbalance_ratio: fraction of positive (Alzheimer's) cases
    """
    rng = np.random.RandomState(seed)
    n_pos = int(n_subjects * imbalance_ratio)
    n_neg = n_subjects - n_pos

    def features(n, label):
        # base means differ between classes to simulate signal
        age = rng.normal(70 if label == 1 else 60, 8, size=n).clip(40, 95)
        fixation_duration_mean = rng.normal(300 if label == 1 else 220, 60, size=n).clip(50, 1000)  # ms
        fixation_count = rng.poisson(40 if label == 1 else 60, size=n).clip(5, 500)
        saccade_length_mean = rng.normal(1.8 if label == 1 else 2.5, 0.6, size=n).clip(0.1, 10)  # degrees
        pupil_diameter_mean = rng.normal(3.0 if label == 1 else 3.5, 0.4, size=n).clip(1.0, 7.0)  # mm
        scanpath_length = fixation_count * saccade_length_mean * rng.uniform(0.8, 1.3, size=n)
        sex = rng.choice(["M", "F"], size=n)
        return pd.DataFrame({
            "age": age,
            "sex": sex,
            "fixation_duration_mean": fixation_duration_mean,
            "fixation_count": fixation_count,
            "saccade_length_mean": saccade_length_mean,
            "pupil_diameter_mean": pupil_diameter_mean,
            "scanpath_length": scanpath_length,
            "label": int(label)
        })

    df_pos = features(n_pos, label=1)
    df_neg = features(n_neg, label=0)
    df = pd.concat([df_pos, df_neg], ignore_index=True).sample(frac=1.0, random_state=seed).reset_index(drop=True)
    df["subject_id"] = [f"S{10000+i}" for i in range(len(df))]
    return df


def load_data(path=DATA_PATH):
    if os.path.exists(path):
        df = pd.read_csv(path)
        print(f"Loaded {len(df)} rows from {path}")
    else:
        print(f"No file at {path} — generating synthetic dataset for demo.")
        df = generate_synthetic_eye_data(n_subjects=1000, imbalance_ratio=0.15)
    return df


def preprocess(df, label_col="label", id_cols=None):
    if id_cols is None:
        id_cols = ["subject_id"]

    # Basic checks
    if label_col not in df.columns:
        raise ValueError(f"Label column '{label_col}' not found in data")

    # Drop id columns
    X = df.drop(columns=[label_col] + [c for c in id_cols if c in df.columns])
    y = df[label_col].astype(int)

    # Simple imputation for numeric, encode categorical
    numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = X.select_dtypes(include=["object", "category"]).columns.tolist()

    # Impute numeric with median
    num_imputer = SimpleImputer(strategy="median")
    X_num = pd.DataFrame(num_imputer.fit_transform(X[numeric_cols]), columns=numeric_cols)

    # For categorical, simple get_dummies after filling NAs
    X_cat = X[cat_cols].fillna("missing")
    X_cat = pd.get_dummies(X_cat, drop_first=True)

    X_processed = pd.concat([X_num.reset_index(drop=True), X_cat.reset_index(drop=True)], axis=1)

    print(f"Preprocessed: {len(numeric_cols)} numeric cols, {len(cat_cols)} categorical cols -> {X_processed.shape[1]} features")
    return X_processed, y


def train_and_evaluate(X, y, use_smote=False):
    # Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.2, random_state=RANDOM_STATE)

    # Optionally apply SMOTE to training set (uncomment import earlier if using)
    if use_smote:
        print("Applying SMOTE to training set...")
        sm = SMOTE(random_state=RANDOM_STATE)
        X_train, y_train = sm.fit_resample(X_train, y_train)
        print("After SMOTE, class counts:", np.bincount(y_train))

    # Pipeline: scaler (not required for RF but keeps alternatives simple) + classifier
    pipeline = Pipeline([
        ("scaler", StandardScaler()),  # tree based models don't require scaling but it's harmless
        ("clf", RandomForestClassifier(random_state=RANDOM_STATE, n_jobs=-1, class_weight="balanced"))
    ])

    # Grid search (small grid; expand when you have compute)
    param_grid = {
        "clf__n_estimators": [100, 300],
        "clf__max_depth": [None, 10, 20],
        "clf__min_samples_split": [2, 5]
    }
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    grid = GridSearchCV(pipeline, param_grid, scoring="roc_auc", cv=cv, verbose=1)
    grid.fit(X_train, y_train)

    print("Best params:", grid.best_params_)
    best_model = grid.best_estimator_

    # Predictions
    y_pred = best_model.predict(X_test)
    y_proba = best_model.predict_proba(X_test)[:, 1] if hasattr(best_model, "predict_proba") else None

    # Metrics
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    roc = roc_auc_score(y_test, y_proba) if y_proba is not None else float("nan")

    print("\nEvaluation on test set")
    print("---------------------")
    print(f"Accuracy:  {acc:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall:    {rec:.4f}")
    print(f"F1:        {f1:.4f}")
    print(f"ROC AUC:   {roc:.4f}")
    print("\nClassification report:\n", classification_report(y_test, y_pred, zero_division=0))
    print("Confusion matrix:\n", confusion_matrix(y_test, y_pred))

    # Feature importances (works because we didn't wrap the RF in custom class)
    # Need to access step 'clf' inside pipeline
    rf = best_model.named_steps["clf"]
    feature_names = X.columns
    importances = rf.feature_importances_
    feat_imp = pd.DataFrame({"feature": feature_names, "importance": importances}).sort_values("importance", ascending=False)
    print("\nTop 10 feature importances:")
    print(feat_imp.head(10).to_string(index=False))

    return best_model, feat_imp, (X_test, y_test, y_pred, y_proba)


def save_model(model, path=MODEL_OUT):
    joblib.dump(model, path)
    print(f"Saved model to {path}")


def main():
    df = load_data(DATA_PATH)
    print(df.head())

    X, y = preprocess(df, label_col="label", id_cols=["subject_id"])
    model, feat_imp, preds = train_and_evaluate(X, y, use_smote=False)

    save_model(model)
    # Optionally: export feature importances to CSV
    feat_imp.to_csv("feature_importances.csv", index=False)
    print("Feature importances saved to feature_importances.csv")


if __name__ == "__main__":
    main()
