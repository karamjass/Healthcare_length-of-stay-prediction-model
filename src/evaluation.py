import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score
)

# ===============================
# Evaluate Single Model
# ===============================
def evaluate_model(model, X_test, y_test, model_name="Model"):
    """
    Prints:
    - Accuracy
    - Weighted F1 score
    - Classification Report
    Shows:
    - Confusion Matrix
    """

    # Predictions
    y_pred = model.predict(X_test)

    # Metrics
    accuracy = accuracy_score(y_test, y_pred)
    weighted_f1 = f1_score(y_test, y_pred, average="weighted")

    # Print metrics
    print("\n" + "="*50)
    print(f"🔹 MODEL: {model_name}")
    print("="*50)
    print("Accuracy       :", round(accuracy, 4))
    print("Weighted F1    :", round(weighted_f1, 4))

    print("\n📊 Classification Report:")
    print(classification_report(y_test, y_pred))

    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)

    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, cmap="Blues", annot=False)
    plt.title(f"Confusion Matrix - {model_name}")
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.tight_layout()
    plt.show()

    return accuracy, weighted_f1


# ===============================
# Evaluate Multiple Models
# ===============================
def evaluate_all_models(models_dict, X_test, y_test):
    """
    models_dict format:
    {
        "Logistic Regression": lr_model,
        "Random Forest": rf_model,
        "XGBoost": xgb_model,
        "CatBoost": cb_model
    }
    """

    results = {}

    for name, model in models_dict.items():
        acc, f1 = evaluate_model(model, X_test, y_test, name)
        results[name] = {
            "Accuracy": acc,
            "Weighted_F1": f1
        }

    return results


# ===============================
# Test Run (OPTIONAL)
# ===============================
if __name__ == "__main__":
    print("⚠️ Run evaluation from modeling.py after training models")
