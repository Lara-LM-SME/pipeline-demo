# CI Pipeline with GitHub Actions & MLflow (Ungraded, At‑Home)

**Required files:** `requirements.txt`, `train.py`, `.github/workflows/main.yml`, and `data/reviews.csv` (500‑row synthetic set from the lesson).

In this at‑home micro‑lab GitHub Actions runs your training script; the script logs to **MLflow** and **exits 0/1** to pass/fail a validation gate.

---

## 1) Create your repo
1. Create a **public** repo named `cicd-ml-pipeline` on GitHub.
2. Clone it locally, copy the lab files in, then:
   ```bash
   git add .
   git commit -m "init: CI gate + MLflow"
   git branch -M main
   git remote add origin https://github.com/YOUR_USERNAME/cicd-ml-pipeline.git
   git push -u origin main
   ```
3. Open the **Actions** tab → watch the run.

> Tip: Add branch protection in GitHub → **Require status checks to pass** so failed CI blocks merges.

---

## 2) Local environment (WSL/Linux/macOS/PowerShell)
> Use a **per‑repo venv** so your notebook/kernel and CI match.

### A) Create & activate a virtual environment
**WSL/Linux/macOS:**
```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
```

**Windows PowerShell:**
```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
```

### B) Install project dependencies
```bash
pip install -r requirements.txt
```

### C) (Notebooks) Install & register a Jupyter kernel
```bash
python -m pip install ipykernel
python -m ipykernel install --user --name cicd-ml --display-name "Python (cicd-ml)"
```
You should see something like:
```
Installed kernelspec cicd-ml in /home/<user>/.local/share/jupyter/kernels/cicd-ml
```
In VS Code/Jupyter, select kernel **Python (cicd-ml)**.

---

## 3) Run locally (optional check)
```bash
# Expected PASS with the full dataset
export DATA_PATH=data/reviews.csv      # (PowerShell: $env:DATA_PATH="data/reviews.csv")
python train.py 1.0

# Force a FAIL for the demo (very strong regularization)
python train.py 1e-8
```

- **`C`** is the model **flexibility** knob (larger = more flexible, smaller = simpler).  
- The **validation gate** lives in `train.py`: if the metric is below the threshold it calls `sys.exit(1)`, making CI fail.

**Artifacts:** MLflow logs to `./mlruns/`. The workflow uploads this folder as `mlflow-run-results` so you can download metrics & the model from the run page.

---

## 4) Trigger CI
Push any change to `main` (or open a PR if your workflow triggers on PRs):
```bash
git commit -am "try: C=1e-8 to demo a fail"
git push
```
Then open **Actions** → inspect logs. You should see:
```
Model Accuracy: ...
Validation Passed | Validation Failed
```

---

## 5) Optional: tiny dataset to illustrate underfitting/variance
If you also include a small CSV like `data/reviews_tiny10.csv` (10 rows), you can flip sources without editing code:
```bash
export DATA_PATH=data/reviews_tiny10.csv
python train.py 1.0     # often unstable / lower accuracy
```
Use together with `C` to demonstrate **under/overfitting** in CI.

---

## 6) What’s in scope vs. out of scope
- **In scope (CI):** clean VM → install deps → run `train.py` → log to MLflow → gate on metric → pass/fail.
- **Out of scope (CD):** model registry & deployment. In a real project, a green run would register/version the model and a separate deploy job would push to staging/prod.

---

## 7) Troubleshooting
- **CI can’t find deps:** ensure `requirements.txt` includes `scikit-learn`, `pandas`, `mlflow` (already provided).
- **Notebook can’t find kernel:** re‑run the `ipykernel` commands above and pick **Python (cicd-ml)**.
- **Git won’t ignore venv:** ensure `.gitignore` includes `.venv/` and commit the `.gitignore` before adding the venv.

---

## 8) Files expected in your repo
```
cicd-ml-pipeline/
├─ data/
│  └─ reviews.csv
├─ .github/
│  └─ workflows/
│     └─ main.yml
├─ requirements.txt
├─ train.py
└─ README.md  ← this file
```
