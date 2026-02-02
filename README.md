#  Breast Cancer Detection Using Support Vector Machine (SVM)

## 📌 Project Overview

This project focuses on building a **machine learning classification model** to detect whether a breast tumor is **malignant or benign** using the **Sklearn Breast Cancer dataset**. Support Vector Machine (SVM) algorithms with different kernels are applied and optimized using **GridSearchCV** for better performance.

The trained model is evaluated using standard classification metrics and saved for future reuse.

##  Dataset Information

* Kaggle Breast Cancer Wisconsin Dataset

##  Tools & Technologies Used

* Python
* Scikit-learn
* Matplotlib
* NumPy
* Pandas
* Joblib

---

##  Steps Performed 

### **Step 1: Load the Dataset**

* Loaded the CSV file (`breast-cancer.csv`) using **Pandas**.
* Inspected the first few rows and dataset shape to understand the structure.
* Checked for missing values to ensure data quality.

### **Step 2: Feature Selection & Target Extraction**

* Separated **features** (`X`) and **target labels** (`y`).
* Dropped irrelevant columns such as `id`.
* Encoded target labels `M` and `B` into numeric values using **LabelEncoder** (`0 = Benign`, `1 = Malignant`).

### **Step 3: Train-Test Split**

* Split the dataset into **training** and **testing sets** to evaluate model generalization.
* Used 80% for training, 20% for testing, with stratification to maintain class distribution.

### **Step 4: Feature Scaling**

* Applied **StandardScaler** to normalize feature values.
* Scaling is critical for SVM as it is sensitive to feature magnitude.

### **Step 5: Baseline SVM Model (Linear Kernel)**

* Trained a **baseline SVM** with a linear kernel to measure initial performance.
* Evaluated using **accuracy score**.

### **Step 6: SVM with RBF Kernel**

* Trained SVM using the **RBF (Radial Basis Function) kernel** to capture non-linear relationships.
* Compared accuracy with the linear model.

### **Step 7: Hyperparameter Tuning with GridSearchCV**

* Created a **pipeline** including `StandardScaler` and `SVC` to simplify preprocessing + model training.
* Tuned **C** (regularization) and **gamma** (kernel coefficient) using **GridSearchCV** for best model.

### **Step 8: Model Evaluation**

* Predicted test set labels using **best SVM model**.
* Generated **confusion matrix** and **classification report** (precision, recall, F1-score).

### **Step 9: ROC Curve & AUC Score**

* Predicted **probabilities** for the positive class.
* Calculated **ROC curve** and **AUC score** to evaluate model discrimination capability.

### **Step 10: Save Trained Pipeline**

* Saved the **pipeline including scaler + tuned SVM** using **Joblib**.
* This allows future predictions without retraining.

---

##  Model Evaluation Metrics

The model performance is evaluated using:

* Accuracy Score
* Precision
* Recall
* F1-Score
* Confusion Matrix
* ROC Curve
* AUC Score

These metrics ensure reliable evaluation for medical classification problems.
