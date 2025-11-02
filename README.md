# CI Pipeline with GitHub Actions & MLflow (Ungraded, At-Home)

**Required files from the brief:** `requirements.txt`, `train.py`, `.github/workflows/main.yml`.

## Quick Start
1. Create a **public** repo `cicd-ml-pipeline` on GitHub and clone it.
2. Copy these files into the repo.
3. Commit & push to `main`:
   ```bash
   git add .
   git commit -m "init"
   git branch -M main
   git remote add origin https://github.com/YOUR_USERNAME/cicd-ml-pipeline.git
   git push -u origin main
   ```
4. Open the **Actions** tab → watch the run.
5. On success, download the **mlflow-run-results** artifact to see metrics and artifacts.

## Local Test (optional)
```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
python train.py 0.5   # likely pass
python train.py 0.01  # likely fail
```
