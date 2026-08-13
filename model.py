import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pickle

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score
from xgboost import XGBClassifier
from sklearn.model_selection import GridSearchCV

# Load dataset
df = pd.read_csv("heart_cleveland_upload.csv")

# Basic EDA
#print(df.head())
#print(df.info())
#print(df.isnull().sum())

# Visualization
#sns.countplot(x='condition', data=df)
#plt.show()

#plt.figure(figsize=(10,8))
#sns.heatmap(df.corr(), annot=True)
#plt.show()

# Features & Target
X = df.drop("condition", axis=1)
y = df["condition"]

# Train test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)




param_grid = {
    'n_estimators': [50, 100, 150],
    'max_depth': [3, 4, 5],
    'learning_rate': [0.01, 0.1, 0.2]
}

grid = GridSearchCV(XGBClassifier(eval_metric='logloss'), param_grid, cv=5)
grid.fit(X_train, y_train)

#print(grid.best_params_)


# Random Forest Model


final_model = XGBClassifier(
    learning_rate=0.1,
    max_depth=5,
    n_estimators=50,
    eval_metric='logloss',
    random_state=42
)

final_model.fit(X_train, y_train)

# Prediction
y_pred = final_model.predict(X_test)
y_prob = final_model.predict_proba(X_test)[:,1]

# Accuracy
print("Accuracy:", accuracy_score(y_test, y_pred))
print("ROC-AUC:", roc_auc_score(y_test, y_prob))
print(classification_report(y_test, y_pred))

# Save model
pickle.dump(final_model, open("model.pkl", "wb"))