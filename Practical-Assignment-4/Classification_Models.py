
import pandas as pd

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    log_loss
)

# Load dataset
data = load_breast_cancer()
X = data.data
y = data.target

# Split dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)

# Create classification models
models = {
    "SVM": make_pipeline(
        StandardScaler(),
        SVC(kernel="rbf", probability=True, random_state=42)
    ),
    "Decision Tree": DecisionTreeClassifier(random_state=42),
    "KNN": make_pipeline(
        StandardScaler(),
        KNeighborsClassifier(n_neighbors=5)
    ),
    "Naive Bayes": GaussianNB()
}

results = []

# Train and evaluate each model
for name, model in models.items():
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)

    results.append({
        "Algorithm": name,
        "Accuracy": round(accuracy_score(y_test, y_pred), 4),
        "Precision": round(precision_score(y_test, y_pred), 4),
        "Recall": round(recall_score(y_test, y_pred), 4),
        "F1-Score": round(f1_score(y_test, y_pred), 4),
        "Log Loss": round(log_loss(y_test, y_prob), 4)
    })

# Display comparison table
df = pd.DataFrame(results)

print("\nCLASSIFICATION ALGORITHM COMPARISON")
print("-" * 85)
print(df.to_string(index=False))

# Display the best results for selected metrics
print("\nPERFORMANCE SUMMARY")
print("Highest Accuracy:", df.loc[df["Accuracy"].idxmax(), "Algorithm"])
print("Highest F1-Score:", df.loc[df["F1-Score"].idxmax(), "Algorithm"])
print("Lowest Log Loss:", df.loc[df["Log Loss"].idxmin(), "Algorithm"])

print("\nProgram executed successfully!")
