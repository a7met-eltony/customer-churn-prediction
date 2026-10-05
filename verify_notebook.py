import json
from pathlib import Path

p = Path(r"d:\Huawei & NTI\project\project.ipynb")
nb = json.loads(p.read_text(encoding="utf-8"))
for i, cell in enumerate(nb["cells"], 1):
    src = cell.get("source", "")
    if isinstance(src, list):
        src = "".join(src)
    if "telco_churn.csv" in src or "pd.read_csv" in src:
        print(f"CELL {i}")
        print(src)
        print("---")
