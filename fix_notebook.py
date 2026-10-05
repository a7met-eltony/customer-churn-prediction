import json
from pathlib import Path

p = Path(r"d:\Huawei & NTI\project\project.ipynb")
nb = json.loads(p.read_text(encoding="utf-8"))

cell_1 = [
    "import pandas as pd\n",
    "import numpy as np\n",
    "from pathlib import Path\n",
    "\n",
    "project_dir = next(\n",
    "    (p for p in [Path.cwd(), Path(r\"d:\\\\Huawei & NTI\\\\project\")] if (p / \"data\" / \"telco_churn.csv\").exists()),\n",
    "    Path.cwd(),\n",
    ")\n",
    "data_path = project_dir / \"data\" / \"telco_churn.csv\"\n",
    "df = pd.read_csv(data_path)\n",
    "df\n",
]
nb["cells"][1]["source"] = cell_1

cell_18 = [
    "from pathlib import Path\n",
    "\n",
    "project_dir = next(\n",
    "    (p for p in [Path.cwd(), Path(r\"d:\\\\Huawei & NTI\\\\project\")] if (p / \"data\" / \"telco_churn.csv\").exists()),\n",
    "    Path.cwd(),\n",
    ")\n",
    "output_dir = project_dir / \"outputs\"\n",
    "output_dir.mkdir(parents=True, exist_ok=True)\n",
    "output_path = output_dir / \"dl_training_curves.png\"\n",
]
nb["cells"][18]["source"] = cell_18

p.write_text(json.dumps(nb, indent=1, ensure_ascii=False), encoding="utf-8")
print("Notebook fixed.")
