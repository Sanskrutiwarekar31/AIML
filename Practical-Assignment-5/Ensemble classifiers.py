import warnings, json
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import StackingClassifier, AdaBoostClassifier, VotingClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, roc_auc_score, confusion_matrix)

RANDOM_STATE = 42

# 1. Load data and split (stratified 80/20)
data = load_breast_cancer()
X, y = data.data, data.target
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE)
print("Train:", X_train.shape, "Test:", X_test.shape,
      "Class counts:", np.bincount(y))

# 2. Base classifiers (scaling inside pipelines to avoid data leakage)
base_models = [
    ("lr",  make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000, random_state=RANDOM_STATE))),
    ("knn", make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=7))),
    ("dt",  DecisionTreeClassifier(max_depth=5, random_state=RANDOM_STATE)),
    ("svm", make_pipeline(StandardScaler(), SVC(kernel="rbf", probability=True, random_state=RANDOM_STATE))),
    ("nb",  GaussianNB()),
]

# 3. Ensemble models
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

# 3a. Stacking: meta-classifier = Logistic Regression on out-of-fold probabilities
stacking = StackingClassifier(
    estimators=base_models,
    final_estimator=LogisticRegression(max_iter=1000, random_state=RANDOM_STATE),
    cv=cv, stack_method="predict_proba", passthrough=False)

# 3b. AdaBoost with decision stumps (depth 1) and with depth-2 trees
ada_stump = AdaBoostClassifier(
    estimator=DecisionTreeClassifier(max_depth=1, random_state=RANDOM_STATE),
    n_estimators=100, learning_rate=0.5, random_state=RANDOM_STATE)
ada_d2 = AdaBoostClassifier(
    estimator=DecisionTreeClassifier(max_depth=2, random_state=RANDOM_STATE),
    n_estimators=100, learning_rate=0.5, random_state=RANDOM_STATE)

# 3c. Voting: hard (majority) voting and soft voting
hard_vote = VotingClassifier(estimators=base_models, voting="hard")
soft_vote = VotingClassifier(estimators=base_models, voting="soft")

all_models = {}
for name, m in base_models:
    all_models[{"lr":"Logistic Regression","knn":"KNN (k=7)","dt":"Decision Tree (d=5)",
                "svm":"SVM (RBF)","nb":"Gaussian NB"}[name]] = m
all_models["Hard Voting"]            = hard_vote
all_models["Soft Voting"]            = soft_vote
all_models["AdaBoost (stumps)"]      = ada_stump
all_models["AdaBoost (depth-2)"]     = ada_d2
all_models["Stacking (LR meta)"]     = stacking

# 4. Train + evaluate
rows, cms = [], {}
for name, model in all_models.items():
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    if hasattr(model, "predict_proba"):
        auc = roc_auc_score(y_test, model.predict_proba(X_test)[:, 1])
    else:
        auc = np.nan                       # hard voting has no probabilities
    cv_acc = cross_val_score(model, X, y, cv=cv, scoring="accuracy")
    rows.append(dict(Model=name,
        Accuracy=accuracy_score(y_test, pred),
        Precision=precision_score(y_test, pred),
        Recall=recall_score(y_test, pred),
        F1=f1_score(y_test, pred),
        ROC_AUC=auc,
        CV_Mean=cv_acc.mean(), CV_Std=cv_acc.std()))
    cms[name] = confusion_matrix(y_test, pred)
    print(f"done {name}")

res = pd.DataFrame(rows)
print(res.round(4).to_string(index=False))
res.to_csv("results.csv", index=False)
json.dump({k: v.tolist() for k, v in cms.items()}, open("cms.json", "w"))

# meta-learner coefficients
coef = stacking.final_estimator_.coef_[0]
print("Meta coefficients:", {n: round(float(c), 3) for (n, _), c in zip(base_models, coef)})

# 5. Plots
order = res.sort_values("Accuracy")
fig, ax = plt.subplots(figsize=(8, 4.6))
w = 0.2; idx = np.arange(len(res))
for i, met in enumerate(["Accuracy", "Precision", "Recall", "F1"]):
    ax.bar(idx + (i-1.5)*w, res[met], w, label=met)
ax.set_xticks(idx); ax.set_xticklabels(res.Model, rotation=35, ha="right", fontsize=8)
ax.set_ylim(0.85, 1.0); ax.set_ylabel("Score"); ax.legend(ncol=4, fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.38), frameon=False)
ax.set_title("Performance of base and ensemble models (test set)")
plt.tight_layout(); plt.savefig("metrics_bar.png", dpi=170); plt.close()

fig, axes = plt.subplots(2, 5, figsize=(13, 5.6))
for ax, (name, cm) in zip(axes.ravel(), cms.items()):
    ax.imshow(cm, cmap="Blues")
    for (i, j), v in np.ndenumerate(cm):
        ax.text(j, i, v, ha="center", va="center", color="white" if v > cm.max()/2 else "black")
    ax.set_title(name, fontsize=8); ax.set_xticks([0,1]); ax.set_yticks([0,1])
    ax.set_xticklabels(["Mal","Ben"], fontsize=7); ax.set_yticklabels(["Mal","Ben"], fontsize=7)
    ax.set_xlabel("Predicted", fontsize=7); ax.set_ylabel("Actual", fontsize=7)
plt.tight_layout(); plt.savefig("confusion_matrices.png", dpi=170); plt.close()

# 6. Sample-level test cases (individual predictions)
label = {0: "Malignant", 1: "Benign"}
i_mal = int(np.where(y_test == 0)[0][0])
i_ben = int(np.where(y_test == 1)[0][0])
# a sample on which base classifiers disagree
base_preds = np.column_stack([m.predict(X_test) for _, m in
                              [(n, all_models[k]) for n, k in zip(["lr","knn","dt","svm","nb"],
                               ["Logistic Regression","KNN (k=7)","Decision Tree (d=5)","SVM (RBF)","Gaussian NB"])]])
i_dis = int(np.where(base_preds.min(1) != base_preds.max(1))[0][0])
for tag, i in [("malignant", i_mal), ("benign", i_ben), ("disagree", i_dis)]:
    xi = X_test[i:i+1]
    print(f"[{tag}] idx={i} actual={label[y_test[i]]} base={base_preds[i].tolist()}",
          "| Hard:", label[int(hard_vote.predict(xi)[0])],
          "| Ada:", label[int(ada_stump.predict(xi)[0])],
          "| Stack:", label[int(stacking.predict(xi)[0])],
          "| P(benign) stack = %.3f" % stacking.predict_proba(xi)[0, 1])