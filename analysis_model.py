import joblib
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import warnings
warnings.filterwarnings('ignore')
sns.set_theme(style="whitegrid")
df = pd.read_csv('AIML Dataset.csv')

# print("head points: ", df.head())
# print("info--data types: ", df.info())
# print("Columns: ", df.columns)
# print("isFraud value counts: ", df['isFraud'].value_counts())
# print("isFlaggedFraud value counts: ", df['isFlaggedFraud'].value_counts())
# print("Missing values in the data frame: ", df.isnull().sum().sum())
# print("shape of data frame: ", df.shape)
# print(round((df["isFraud"].value_counts(normalize=True))*100, 2))
# print(df["type"].value_counts().plot(kind='bar', title="Transaction Types", color="skyblue"),
#       plt.title("Transaction Types Distribution"),
#       plt.xlabel("Transaction Type"),
#       plt.ylabel("Count"),
#       plt.show())

fraud_by_type = df.groupby(
    "type")["isFraud"].mean().sort_values(ascending=False)
# print(fraud_by_type.plot(
#     kind='bar', title="Fraud Rate by Transaction Type", color="salmon"),
#     plt.title("Fraud Rate by Transaction Type"),
#     plt.xlabel("Transaction Type"),
#     plt.ylabel("Fraud Rate"),
#     plt.show())

# print("Statistical summary of 'amount' column: ",
#       df['amount'].describe().astype(int))

# print(sns.histplot(np.log1p(df['amount']), bins=50, kde=True, color='blue'),
#       plt.title("Transaction Amount Distribution (log scale)"),
#       plt.xlabel("Amount"),
#       plt.show())
# print(df.columns)

# df["balanceDiffOrig"] = df["oldbalanceOrg"] - df["newbalanceOrig"]
# df["balanceDiffDest"] = df["oldbalanceDest"] - df["newbalanceDest"]
# print((df["balanceDiffOrig"] < 0).sum())
# print((df["balanceDiffDest"] < 0).sum())
# print(df.head())

fraud_per_step = df[df["isFraud"] == 1]["step"].value_counts().sort_index()
# print(
#     plt.plot(fraud_per_step.index, fraud_per_step.values,
#              label='Fraudulent Transactions per Step', color='red'),
#     plt.xlabel("Step(Time)"),
#     plt.ylabel("Number of Frauds"),
#     plt.title("Frauds Over Time"),
#     plt.grid(True),
#     plt.show())

# print(df.head())
# print(df.drop(columns="step", inplace=True))
# print(df.head())

# df_top_senders = df["nameOrig"].value_counts().head(10)
# print(df_top_senders)

# df_top_receivers = df["nameDest"].value_counts().head(10)
# print(df_top_receivers)

# fraud_users = df[df["isFraud"] == 1]["nameOrig"].value_counts().head(10)
# print(fraud_users)

# print(df.head())
fraud_types = df[df["type"].isin(
    ["TRANSFER", "CASH_OUT"])]
# print(fraud_types)

# sns.countplot(data=fraud_types, x="type", hue="isFraud")
# plt.title("Fraudulent Transactions by Type")
# plt.show()
corr = df[["amount", "oldbalanceOrg", "newbalanceOrig",
           "oldbalanceDest", "newbalanceDest", "isFraud"]].corr()
# print(sns.heatmap(corr, annot=True, cmap='cool
# print(corr)

# sns.heatmap(corr, annot=True, cmap='coolwarm')
# plt.title("Correlation Matrix")
# plt.show()

zero_after_transfer = df[
    (df["oldbalanceOrg"] > 0) &
    (df["newbalanceOrig"] == 0) &
    (df["type"].isin(["TRANSFER", "CASH_OUT"]))
]

len_zero_after_transfer = len(zero_after_transfer)
# print("Number of transactions with zero new balance after transfer/cash out: ",
#       len_zero_after_transfer)

# print(zero_after_transfer.head())

# df["isFraud"].value_counts(normalize=True)


# print(df.head())

df_model = df.drop(["nameOrig", "nameDest", "isFlaggedFraud"], axis=1)
# print(df_model.head())

categorical_features = ["type"]
numerical_features = ["amount", "oldbalanceOrg",
                      "newbalanceOrig", "oldbalanceDest", "newbalanceDest"]

y = df_model["isFraud"]
X = df_model.drop("isFraud", axis=1)

x_train, x_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, stratify=y)

preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numerical_features),
        ('cat', OneHotEncoder(drop="first"), categorical_features)
    ],
    remainder="drop"
)

pipeline = Pipeline([
    ("prep", preprocessor),
    ("clf", LogisticRegression(class_weight="balanced", max_iter=1000))
])

pipeline.fit(x_train, y_train)
y_pred = pipeline.predict(x_test)
print(classification_report(y_test, y_pred))

cm = confusion_matrix(y_test, y_pred)
print("Confusion Matrix: ", cm)

pipeline_score = pipeline.score(x_test, y_test) * 100  # accuracy score
print("Pipeline Accuracy Score: ", pipeline_score)

joblib.dump(pipeline, 'fraud_detection_model.pkl')
