import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

def check_metrics():
    print("Loading model...")
    model = joblib.load('mental_health_model.joblib')

    print("Loading dataset...")
    df = pd.read_csv('mental_health_dataset.csv')
    
    TARGET = df.columns[-1] 
    X = df.drop(columns=[TARGET])
    y = df[TARGET]

    # Using the same random state to match training split
    _, X_test, _, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    print("Evaluating model...\n")
    y_pred = model.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {acc * 100:.2f}%\n")

    print("Classification Report:")
    print(classification_report(y_test, y_pred))

    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))

if __name__ == "__main__":
    check_metrics()