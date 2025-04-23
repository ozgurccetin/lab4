# Лабораторная 4 — Ансамбли и полносвязные нейросети
# Установка:
# !pip install imbalanced-learn
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import roc_curve, roc_auc_score, auc
from imblearn.over_sampling import SMOTE

# Загрузка данных
df = pd.read_csv("german.csv", sep=";")
X = df.drop("Creditability", axis=1)
y = df["Creditability"]

X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.2, random_state=42)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# SMOTE для балансировки
smote = SMOTE(random_state=42)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train_scaled, y_train)

# Сравниваемые модели
models = {
    "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
    "Gradient Boosting": GradientBoostingClassifier(n_estimators=100, random_state=42),
    "MLPClassifier (extreme boosted)": MLPClassifier(
        hidden_layer_sizes=(512, 256, 128, 64),
        activation='relu',
        learning_rate_init=0.002,
        alpha=1e-5,
        solver='adam',
        max_iter=2000,
        early_stopping=True,
        n_iter_no_change=20,
        random_state=42,
        verbose=False
    )
}

# ROC-кривые и AUC
plt.figure(figsize=(10, 6))
auc_scores = {}
for name, model in models.items():
    if "MLP" in name:
        model.fit(X_train_resampled, y_train_resampled)
        y_proba = model.predict_proba(X_test_scaled)[:, 1]
    else:
        model.fit(X_train, y_train)
        y_proba = model.predict_proba(X_test)[:, 1]

    fpr, tpr, _ = roc_curve(y_test, y_proba)
    score = roc_auc_score(y_test, y_proba)
    auc_scores[name] = score
    plt.plot(fpr, tpr, lw=2, label=f"{name} (AUC = {score:.3f})")

plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC-кривые всех моделей")
plt.legend(loc="lower right")
plt.grid(True)
plt.show()

# Bar-график AUC
plt.figure(figsize=(7, 4))
plt.barh(list(auc_scores.keys()), list(auc_scores.values()), color='skyblue')
plt.xlabel("ROC AUC")
plt.title("Сравнение моделей по AUC")
for i, (model, score) in enumerate(auc_scores.items()):
    plt.text(score + 0.005, i, f"{score:.3f}", va='center')
plt.xlim(0.7, 1.0)
plt.grid(True, axis='x')
plt.tight_layout()
plt.show()

# Вывод результатов
print("\nИтоговые значения ROC AUC:")
for model, score in auc_scores.items():
    print(f"{model}: {score:.4f}")