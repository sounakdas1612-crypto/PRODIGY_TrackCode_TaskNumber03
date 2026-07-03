
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import plot_tree
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report

df = pd.read_csv("/content/bank-full.txt", sep=';')

print("First 5 Rows")
print(df.head())

print("\nDataset Shape:")
print(df.shape)

print("\nDataset Info:")
print(df.info())

encoder = LabelEncoder()

for column in df.columns:
    if df[column].dtype == "object":
        df[column] = encoder.fit_transform(df[column])

X = df.drop("y", axis=1)
y = df["y"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

model = DecisionTreeClassifier(random_state=42)

model.fit(X_train, y_train)

y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)

print("\nAccuracy:", accuracy)

print("\nConfusion Matrix")

print(confusion_matrix(y_test, y_pred))

print("\nClassification Report")

print(classification_report(y_test, y_pred))

plt.figure(figsize=(22,10))

plot_tree(
    model,
    filled=True,
    feature_names=X.columns,
    class_names=["No","Yes"],
    fontsize=8
)

plt.title("Decision Tree Classifier")

plt.show()

from sklearn.tree import plot_tree

plt.figure(figsize=(18,10))

plot_tree(
    model,
    max_depth=3,
    filled=True,
    feature_names=X.columns,
    class_names=["No","Yes"],
    fontsize=10
)

plt.show()
