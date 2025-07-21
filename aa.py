# Peace, Justice, and Strong Institutions – Machine Learning Project
# Simulating classification of high/low institutional risk

# STEP 1: Import Libraries
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report
import matplotlib.pyplot as plt
import seaborn as sns

# STEP 2: Simulate Dataset (replace with real data if available)
# Features: crime_rate, corruption_index, trust_in_government, justice_access_score, human_rights_score
np.random.seed(42)
data_size = 500
df = pd.DataFrame({
    'crime_rate': np.random.uniform(0, 100, data_size),
    'corruption_index': np.random.uniform(0, 10, data_size),
    'trust_in_government': np.random.uniform(0, 100, data_size),
    'justice_access_score': np.random.uniform(0, 100, data_size),
    'human_rights_score': np.random.uniform(0, 100, data_size),
})

# Create a binary label: 1 = High Institutional Risk, 0 = Low Risk
df['institutional_risk'] = (
    (df['crime_rate'] > 70) |
    (df['corruption_index'] > 7) |
    (df['trust_in_government'] < 30) |
    (df['justice_access_score'] < 40)
).astype(int)

# STEP 3: Split Dataset
X = df.drop('institutional_risk', axis=1)
y = df['institutional_risk']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# STEP 4: Train Model
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

# STEP 5: Make Predictions
y_pred = model.predict(X_test)

# STEP 6: Confusion Matrix and Evaluation
cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Low Risk", "High Risk"])
disp.plot(cmap='Blues')
plt.title("Confusion Matrix – Institutional Risk Classification")
plt.show()

# Classification Report
print("\nClassification Report:\n")
print(classification_report(y_test, y_pred, target_names=["Low Risk", "High Risk"]))

# STEP 7: Feature Importance Plot (optional)
feature_importances = pd.Series(model.feature_importances_, index=X.columns)
plt.figure(figsize=(8, 5))
sns.barplot(x=feature_importances.values, y=feature_importances.index)
plt.title("Feature Importance in Predicting Institutional Risk")
plt.xlabel("Importance Score")
plt.ylabel("Feature")
plt.tight_layout()
plt.show()
