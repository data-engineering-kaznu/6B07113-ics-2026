import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer #nan ср.
from sklearn.preprocessing import StandardScaler 
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

columns = [
    "sepal_length",
    "sepal_width",
    "petal_length",
    "petal_width",
    "class"
]

df = pd.read_csv("iris.data", header=None, names=columns)

print("Первые строки:")
print(df.head())
df["class"] = df["class"].astype("category").cat.codes
X = df.drop("class", axis=1).values
y = df["class"].values
imputer = SimpleImputer(strategy="mean") 
X = imputer.fit_transform(X)

scaler = StandardScaler()
X = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42
)

model = KNeighborsClassifier(n_neighbors=5)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
cm = confusion_matrix(y_test, y_pred)
report = classification_report(y_test, y_pred)

print("\nAccuracy:", accuracy)
print("\nConfusion matrix:\n", cm)
print("\nClassification report:\n", report)
