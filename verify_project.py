from pathlib import Path
import nbformat

ROOT = Path(__file__).resolve().parent

required_files = [
    ROOT / "project.ipynb",
    ROOT / "streamlit_app.py",
    ROOT / "churn_features.py",
    ROOT / "requirements.txt",
    ROOT / "README.md",
    ROOT / "data" / "telco_churn.csv",
]

for path in required_files:
    assert path.exists(), f"Missing required file: {path}"

nb = nbformat.read(ROOT / "project.ipynb", as_version=4)
source = "\n".join(cell.source for cell in nb.cells if cell.cell_type == "code")

checks = {
    "missing values": "fillna" in source,
    "encoding": "OneHotEncoder" in source,
    "scaling": "StandardScaler" in source,
    "train test split": "train_test_split" in source,
    "EDA": "Exploratory Data Analysis" in "\n".join(
        cell.source for cell in nb.cells if cell.cell_type == "markdown"
    ),
    "KNN": "KNeighborsClassifier" in source,
    "Decision Tree": "DecisionTreeClassifier" in source,
    "Random Forest": "RandomForestClassifier" in source,
    "SVM": "SVC" in source,
    "Naive Bayes": "GaussianNB" in source,
    "Deep Learning": "keras.Sequential" in source,
    "validation": "validation_data" in source,
    "loss curve": "dl_training_curves.png" in source,
    "confusion matrix": "confusion_matrices.png" in source,
    "comparison": "comparison" in source,
    "final prediction": "Churn Probability" in source,
    "feature engineering": "add_features" in source,
    "feature selection": "mutual_info_classif" in source,
    "class balancing (SMOTE)": "SMOTE" in source,
    "Streamlit": (ROOT / "streamlit_app.py").exists(),
}

failed = [name for name, ok in checks.items() if not ok]
if failed:
    raise AssertionError("Failed checks: " + ", ".join(failed))

print("Static project verification: PASS")
print(f"Notebook cells: {len(nb.cells)}")
