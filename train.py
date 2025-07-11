# 1. Préparation des données et entraînement du modèle (`train.py`)
import pandas as pd
import numpy
import seaborn as sns
import matplotlib.pyplot as plt
from ydata_profiling import ProfileReport
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import joblib

# Chagement des données
df = pd.read_csv("Expresso_churn_dataset.csv")
print(df.head())

# Exploration basique des données
print("Shape of the DataFrame:", df.shape)
print("\nData types of columns:")
df.info()
print("\nSample of 5 random rows:")
print(df.sample(5))

# Rapport de profilage

profile_report = ProfileReport(df, title="Rapport de Profilage")
#profile_report.to_file("expresso_report.html")
print(profile_report)

# Valeurs manquants

# Drop columns with a high percentage of missing values
df = df.drop(['ZONE1', 'ZONE2'], axis=1)

# Impute missing values in numerical columns with the median
numerical_cols = ['MONTANT', 'FREQUENCE_RECH', 'REVENUE', 'ARPU_SEGMENT', 'FREQUENCE', 'DATA_VOLUME', 'ON_NET', 'ORANGE', 'TIGO', 'REGULARITY', 'FREQ_TOP_PACK']
for col in numerical_cols:
    if df[col].isnull().any():
        median_val = df[col].median()
        df[col] = df[col].fillna(median_val)

# Impute missing values in categorical columns with the mode
categorical_cols = ['REGION', 'MRG', 'TOP_PACK']
for col in categorical_cols:
    if df[col].isnull().any():
        mode_val = df[col].mode()[0]
        df[col] = df[col].fillna(mode_val)

# Verify that all missing values have been handled
print("\nMissing values after handling:")
print(df.isnull().sum())

# Les doublons de valeur

initial_rows = df.shape[0]
df.drop_duplicates(inplace=True)
rows_after_dropping = df.shape[0]

print(f"Number of rows before removing duplicates: {initial_rows}")
print(f"Number of rows after removing duplicates: {rows_after_dropping}")

#
numerical_cols_for_outliers = ['MONTANT', 'FREQUENCE_RECH', 'REVENUE', 'ARPU_SEGMENT', 'FREQUENCE', 'DATA_VOLUME', 'ON_NET', 'ORANGE', 'TIGO', 'REGULARITY', 'FREQ_TOP_PACK']

for col in numerical_cols_for_outliers:
    upper_bound = df[col].quantile(0.99)
    df[col] = df[col].clip(upper=upper_bound)

plt.figure(figsize=(15, 10))
for i, col in enumerate(numerical_cols_for_outliers):
    plt.subplot(4, 3, i + 1)
    sns.boxplot(x=df[col])
    plt.title(col)
plt.tight_layout()
plt.show()

# Encodge ctegoril fetures

categorical_cols_for_encoding = ['REGION', 'TENURE', 'MRG', 'TOP_PACK']

df_encoded = pd.get_dummies(df, columns=categorical_cols_for_encoding, drop_first=True)

df_encoded = df_encoded.drop('user_id', axis=1)

print(df_encoded.head())

# Split d

X = df_encoded.drop('CHURN', axis=1)
y = df_encoded['CHURN']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print("Shape of X_train:", X_train.shape)
print("Shape of X_test:", X_test.shape)
print("Shape of y_train:", y_train.shape)
print("Shape of y_test:", y_test.shape)

# Modelis et Evlution

# Instantiate the classifier
model = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')

# Train the classifier
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)

# Sauvegarde du modèle et des encodeurs
joblib.dump(model, 'churn_model.joblib')
joblib.dump(df_encoded, 'df_encoded.joblib')

print(f"Accuracy: {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1-score: {f1:.4f}")
print("\nConfusion Matrix:")
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', cbar=False)
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.title('Confusion Matrix')
plt.show()