#ALGORITHM NAME - Linear classification using SVM

#IMPORTING REQUIRED LIBRARIES
import numpy as np #to perform numerical operations
from sklearn.datasets import fetch_openml #fetching dataset 
from sklearn.model_selection import train_test_split #splitting data into training and testing data
from sklearn.svm import LinearSVC #
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt

# Step 1: Data Preparation
mnist = fetch_openml("mnist_784")
X, y = mnist.data, mnist.target.astype(int)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
X_train /= 255.0
X_test /= 255.0

# Step 2: Model Creation
svm_classifier = LinearSVC(random_state=42)

# Step 3: Model Training
svm_classifier.fit(X_train, y_train)

# Step 4: Model Evaluation
y_pred = svm_classifier.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)

print(f"Accuracy: {accuracy * 100:.2f}%")
print("Confusion Matrix:")
print(conf_matrix)   

# Visualize some sample predictions
plt.figure(figsize=(10, 6))
for i in range(10):
    plt.subplot(2, 5, i + 1)
    plt.imshow(X_test.iloc[i].to_numpy().reshape(28, 28), cmap='gray')
    plt.title(f"True: {y_test.iloc[i]}, Pred: {y_pred[i]}")
    plt.axis('off')
plt.show() 