# 🧠 Breast Cancer Classification using Machine Learning

---

## a. Problem Statement

The objective of this project is to build and compare multiple machine learning classification models to predict whether a breast tumor is **malignant** or **benign** using diagnostic medical measurements.
Six classification algorithms are implemented and evaluated using multiple performance metrics. The best-performing model is identified based on comparative analysis.

---

## b. Dataset Description  *(1 Mark)*

The dataset used in this project is the **Breast Cancer Wisconsin Dataset** from the UCI Machine Learning Repository.

### 📊 Dataset Characteristics

| Property | Value |
|----------|--------|
| Total Instances | **569** |
| Total Features | **30 numerical features** |
| Target Classes | 0 → Malignant, 1 → Benign |
| Feature Type | All Numerical |
| Missing Values | None |

The features are computed from digitized images of fine needle aspirate (FNA) of breast masses and describe characteristics such as radius, texture, perimeter, area, smoothness, compactness, concavity, symmetry, and fractal dimension.

✔ Minimum 12 features satisfied  
✔ Minimum 500 instances satisfied  
✔ Binary classification problem  

---

## c. Models Used and Performance Comparison  *(6 Marks)*

The following six machine learning classification models were implemented:

1. Logistic Regression  
2. Decision Tree Classifier  
3. K-Nearest Neighbors (kNN)  
4. Gaussian Naive Bayes  
5. Random Forest (Ensemble)  
6. XGBoost (Ensemble)  

All models were trained using an **80:20 stratified train-test split**.

### 📈 Evaluation Metrics Used

- Accuracy  
- AUC Score  
- Precision  
- Recall  
- F1 Score  
- Matthews Correlation Coefficient (MCC)

---

## 🔍 Model Comparison Table

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 Score | MCC |
|---------------|----------|------|-----------|--------|-----------|------|
| **Logistic Regression** | **0.982** | **0.994** | **0.986** | **0.986** | **0.986** | **0.962** |
| Decision Tree | 0.912 | 0.857 | 0.908 | 0.958 | 0.932 | 0.810 |
| kNN | 0.965 | 0.995 | 0.972 | 0.972 | 0.972 | 0.925 |
| Naive Bayes | 0.886 | 0.953 | 0.928 | 0.889 | 0.908 | 0.760 |
| Random Forest (Ensemble) | 0.956 | 0.989 | 0.947 | 0.986 | 0.966 | 0.906 |
| XGBoost (Ensemble) | 0.956 | 0.989 | 0.959 | 0.972 | 0.966 | 0.905 |

---

## 📌 Observations on Model Performance  *(3 Marks)*

| ML Model Name | Observation about Model Performance |
|---------------|-------------------------------------|
| **Logistic Regression** | Achieved the highest accuracy and MCC, indicating the dataset is nearly linearly separable. It generalized extremely well with balanced precision and recall. |
| Decision Tree | Showed lower AUC compared to other models, likely due to overfitting and instability of a single tree model. |
| kNN | Performed very strongly with the highest AUC score, indicating good class separation using distance-based learning. |
| Naive Bayes | Lower performance due to strong feature correlations violating the independence assumption. |
| Random Forest (Ensemble) | Performed well with high recall and strong MCC. Ensemble learning improved robustness over a single tree. |
| XGBoost (Ensemble) | Comparable to Random Forest and achieved strong performance but did not significantly outperform Logistic Regression due to dataset size and structure. |

---

## ✅ Conclusion

**Logistic Regression emerged as the best-performing model**, suggesting that the dataset is largely linearly separable. While ensemble methods performed competitively, they did not significantly outperform simpler models due to the structured nature and relatively small size of the dataset.

---

## 🚀 Deployment

This project is deployed using **Streamlit Community Cloud**.

To run locally:

```bash
pip install -r requirements.txt
streamlit run app.py
