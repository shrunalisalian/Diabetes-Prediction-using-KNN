# 🏥 **Diabetes Prediction using K-Nearest Neighbors (KNN)**  

**Skills:** KNN Algorithm, Machine Learning, Data Preprocessing, Model Evaluation  

---

## 🚀 **Project Overview**  
This project explores **K-Nearest Neighbors (KNN)** to predict whether a patient is diabetic based on **medical parameters**. It is a **beginner-friendly ML project** that covers:  

✅ **Data Preprocessing & Cleaning**  
✅ **Feature Scaling & Standardization**  
✅ **Implementing KNN Classifier using Scikit-Learn**  
✅ **Evaluating Model Performance with Accuracy & F1 Score**  

📌 **Project Reference:** [Simplilearn KNN Tutorial](https://www.youtube.com/watch?v=4HKqjENq9OU)  

---

## 🎯 **Key Objectives**  
✔ **Understand the working of KNN in a classification problem**  
✔ **Explore medical features affecting diabetes risk**  
✔ **Train & evaluate a KNN model on real-world medical data**  
✔ **Optimize KNN performance using hyperparameter tuning**  

---

## 📊 **Dataset Overview: PIMA Indian Diabetes Dataset**  
The dataset contains **medical diagnostic data** from **female patients aged 21+**.  

📌 **Feature Overview:**  
- **Pregnancies** – Number of times pregnant  
- **Glucose** – Blood glucose level  
- **Blood Pressure** – Diastolic blood pressure (mm Hg)  
- **Skin Thickness** – Triceps skinfold thickness (mm)  
- **Insulin** – 2-hour serum insulin (mu U/ml)  
- **BMI** – Body Mass Index  
- **DiabetesPedigreeFunction** – Diabetes likelihood based on family history  
- **Age** – Patient's age  
- **Outcome** – **(Target variable: 0 = No Diabetes, 1 = Diabetes)**  

✅ **Example: Loading the Dataset**  
```python
import pandas as pd

df = pd.read_csv("diabetes.csv")
df.head()
```

✅ **Checking Missing or Invalid Values**  
```python
df.info()
df.describe()
```
💡 **Observation:**  
- Features like **Glucose & Blood Pressure should not have zero values** (need handling).  

✅ **Handling Invalid Values**  
```python
df["Glucose"].replace(0, df["Glucose"].median(), inplace=True)
df["BloodPressure"].replace(0, df["BloodPressure"].median(), inplace=True)
```

---

## 📈 **Exploratory Data Analysis (EDA)**  
We explore **correlations** between features and diabetes.  

✅ **Example: Correlation Heatmap**  
```python
import seaborn as sns
import matplotlib.pyplot as plt

plt.figure(figsize=(8, 6))
sns.heatmap(df.corr(), annot=True, cmap="coolwarm")
plt.title("Feature Correlation Heatmap")
plt.show()
```
💡 **Findings:**  
- **Glucose levels are highly correlated with diabetes.**  
- **BMI & Age also contribute significantly to diabetes risk.**  

✅ **Example: Visualizing Diabetes vs Non-Diabetes Cases**  
```python
sns.countplot(x="Outcome", data=df)
plt.title("Diabetes vs Non-Diabetes Cases")
```
💡 **Insight:**  
- The dataset has a slight **class imbalance** (more non-diabetic patients).  

---

## 🏗 **Feature Engineering & Data Preprocessing**  
Before training our KNN model, we **scale features** to ensure proper distance-based classification.  

✅ **Train-Test Split**  
```python
from sklearn.model_selection import train_test_split

X = df.drop(columns=["Outcome"])
y = df["Outcome"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
```

✅ **Feature Scaling using StandardScaler**  
```python
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
```
💡 **Why?** – KNN is a **distance-based algorithm**, and scaling prevents bias from large-valued features.  

---

## 🤖 **Implementing KNN Model for Diabetes Prediction**  
We train a **KNN classifier** using `sklearn.neighbors.KNeighborsClassifier`.  

✅ **Training the Model**  
```python
from sklearn.neighbors import KNeighborsClassifier

knn = KNeighborsClassifier(n_neighbors=5)  # Default k=5
knn.fit(X_train, y_train)
```

✅ **Making Predictions**  
```python
y_pred = knn.predict(X_test)
```

---

## 📊 **Model Evaluation & Performance Metrics**  
We assess KNN's performance using **accuracy, F1-score, and confusion matrix**.  

✅ **Accuracy Score**  
```python
from sklearn.metrics import accuracy_score

accuracy = accuracy_score(y_test, y_pred)
print(f"KNN Model Accuracy: {accuracy:.2f}")
```

✅ **Confusion Matrix**  
```python
from sklearn.metrics import confusion_matrix

cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
plt.title("Confusion Matrix for KNN Model")
plt.show()
```

✅ **F1-Score Calculation**  
```python
from sklearn.metrics import f1_score

f1 = f1_score(y_test, y_pred)
print(f"KNN Model F1-Score: {f1:.2f}")
```
💡 **Findings:**  
- **Higher accuracy means a good model, but we must also check F1-score to handle class imbalance.**  

---

## 🔍 **Optimizing KNN with Hyperparameter Tuning**  
To improve performance, we **experiment with different values of K**.  

✅ **Finding the Best K Value**  
```python
import numpy as np
from sklearn.model_selection import cross_val_score

k_range = range(1, 21)
accuracy_scores = []

for k in k_range:
    knn = KNeighborsClassifier(n_neighbors=k)
    scores = cross_val_score(knn, X_train, y_train, cv=5, scoring="accuracy")
    accuracy_scores.append(scores.mean())

# Plot results
plt.plot(k_range, accuracy_scores, marker="o")
plt.xlabel("Number of Neighbors (K)")
plt.ylabel("Cross-Validated Accuracy")
plt.title("Optimizing K in KNN")
plt.show()
```
💡 **Result:** The optimal **K value** is found at the **elbow point** in the graph.  

✅ **Training the Model with Optimized K**  
```python
best_k = np.argmax(accuracy_scores) + 1  # Since index starts from 0
knn_optimized = KNeighborsClassifier(n_neighbors=best_k)
knn_optimized.fit(X_train, y_train)
```

---

## 🔮 **Future Enhancements**  
🔹 **Compare KNN with Logistic Regression & Decision Trees**  
🔹 **Use GridSearchCV for automated hyperparameter tuning**  
🔹 **Apply PCA for dimensionality reduction before training**  

---

## 🎯 **Why This Project Stands Out for ML & AI Roles**  
✔ **Explains KNN in a step-by-step manner for beginners**  
✔ **Applies Data Preprocessing & Feature Engineering for real-world datasets**  
✔ **Optimizes KNN using Cross-Validation for better accuracy**  
✔ **Evaluates Model Performance with Accuracy & F1-Score**  

---

## 🛠 **How to Run This Project**  
1️⃣ Clone the repo:  
   ```bash
   git clone https://github.com/shrunalisalian/diabetes-prediction-knn.git
   ```
2️⃣ Install dependencies:  
   ```bash
   pip install -r requirements.txt
   ```
3️⃣ Run the Jupyter Notebook:  
   ```bash
   jupyter notebook "Predicting Diabetes using K-Nearest Neighbors.ipynb"
   ```

---

## 📌 **Connect with Me**  
- **LinkedIn:** [Shrunali Salian](https://www.linkedin.com/in/shrunali-salian/)  
- **Portfolio:** [Your Portfolio Link](#)  
- **Email:** [Your Email](#)  
