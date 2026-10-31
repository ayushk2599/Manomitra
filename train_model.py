import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

def train():
    print("Loading dataset...")
    df = pd.read_csv('mental_health_dataset.csv')
    
    # Assuming the target variable is the last column
    TARGET = df.columns[-1] 
    X = df.drop(columns=[TARGET])
    y = df[TARGET]

    # Identify numerical and categorical columns dynamically
    numeric_features = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
    categorical_features = X.select_dtypes(include=['object', 'category']).columns.tolist()

    print("Setting up preprocessing pipelines...")
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numeric_features),
            ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
        ])

    print("Training models...")
    models = {
        "Logistic Regression": Pipeline([
            ('pre', preprocessor),
            ('clf', LogisticRegression(max_iter=1000, class_weight='balanced'))
        ]),
        "Random Forest": Pipeline([
            ('pre', preprocessor),
            ('clf', RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=42))
        ])
    }

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    best_model = None
    best_acc = 0
    best_name = ""

    for name, pipeline in models.items():
        pipeline.fit(X_train, y_train)
        y_pred = pipeline.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        print(f"{name} Accuracy: {acc:.4f}")
        
        if acc > best_acc:
            best_acc = acc
            best_model = pipeline
            best_name = name

    print(f"\nBest Model Selected: {best_name} with Accuracy: {best_acc:.4f}")

    # Save the best model
    joblib.dump(best_model, 'mental_health_model.joblib')
    print("Model saved successfully as 'mental_health_model.joblib'")

if __name__ == "__main__":
    train()