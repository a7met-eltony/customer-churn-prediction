import json
from pathlib import Path

p = Path(r"d:\Huawei & NTI\project\project.ipynb")
nb = json.loads(p.read_text(encoding="utf-8"))
for idx, cell in enumerate(nb["cells"]):
    src = cell.get("source", "")
    if isinstance(src, list):
        src = "".join(src)
    if any(k in src for k in ["telco_churn.csv", "preds = {}", "results = pd.DataFrame", "model_comparison", "output_dir", "pd.read_csv"]):
        print(f"CELL {idx}")
        print(src)
        print("---")
