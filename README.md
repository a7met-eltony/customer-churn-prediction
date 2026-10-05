# AI Project — Customer Churn Prediction

An end-to-end AI system that predicts whether a telecom customer will **Churn** or **Not Churn** using Machine Learning and Deep Learning.

## Project Requirements Covered

- Data preprocessing
  - Missing values
  - Categorical encoding
  - Feature scaling
  - Feature engineering (7 new features)
  - Feature selection (Mutual Information)
  - Class balancing with SMOTE (training data only)
  - Train/Test split
- EDA
  - Churn distribution
  - Tenure vs Churn
  - Monthly Charges vs Churn
  - Categorical churn-rate analysis
  - Correlation heatmap
- Machine Learning
  - KNN
  - Decision Tree
  - Random Forest
  - SVM
  - Naive Bayes
- Deep Learning
  - TensorFlow/Keras neural network
  - Training and validation
  - Loss and accuracy curves
  - SMOTE-balanced training data
- Evaluation
  - Accuracy
  - Precision
  - Recall
  - F1-Score
  - ROC-AUC
  - Confusion matrices
  - ML vs Deep Learning comparison
- Final prediction
  - Churn / No Churn
  - Churn probability
- Streamlit interface
- Saved model artifacts and visualizations

## Project Structure

```text
project/
├── data/
│   └── telco_churn.csv
├── models/
│   ├── preprocessor.joblib
│   ├── model_selection.joblib
│   ├── knn.joblib
│   ├── decision_tree.joblib
│   ├── random_forest.joblib
│   ├── svm.joblib
│   ├── naive_bayes.joblib
│   └── deep_learning_model.keras
├── outputs/
│   ├── EDA visualizations
│   ├── dl_training_curves.png
│   ├── confusion_matrices.png
│   ├── metrics_comparison.png
│   └── model_comparison.csv
├── project.ipynb
├── churn_features.py
├── streamlit_app.py
├── requirements.txt
└── README.md
```

## Run the Notebook

```bash
pip install -r requirements.txt
jupyter notebook project.ipynb
```

Run all notebook cells from top to bottom. The notebook creates the model artifacts and output visualizations used by Streamlit.

## Run Streamlit

```bash
streamlit run streamlit_app.py
```

The app provides:
1. New-customer churn prediction with probability.
2. Model comparison.
3. Confusion matrices and DL training curves.
4. EDA visualizations.

## Handling Imbalance and Features

- The target is imbalanced (~73% No Churn / ~27% Churn). **SMOTE** is applied to the **training data only** (inside an `imblearn` Pipeline for ML models, and on the training split for the neural network). The validation and test sets keep the real distribution.
- `churn_features.py` creates engineered features: `num_addon_services`, `num_protection_services`, `avg_monthly_spend`, `charge_vs_avg_ratio`, `tenure_group`, `is_autopay`, `has_internet`.
- Features with near-zero Mutual Information with Churn (`gender`, `PhoneService`, `MultipleLines`) are removed.

## Model Selection

The notebook selects the best model using **F1-Score**, because churn prediction is a binary classification problem where balancing false positives and false negatives is important.

## Notes

The dataset is the Telco Customer Churn dataset. `TotalCharges` is converted to numeric and missing values are handled before modeling. The identifier `customerID` is removed because it is not a meaningful predictive feature.
