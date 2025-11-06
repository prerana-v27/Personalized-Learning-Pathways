# ml/data README

This folder (and the `ml/` package) reads processed datasets from the repository's data directory.

Where the script looks for data
- By default `ml/train.py` uses the environment variable `REPO_DATA_DIR` to find the data directory.
- If `REPO_DATA_DIR` is not set, it defaults to the repository sibling `../Data` (i.e. `Data/processed`).

Expected files
- The training script looks for the following files under `<REPO_DATA_DIR>/processed/`:
  - `coursera_cleaned.csv`
  - `edx_cleaned.csv`
  - `nptel_cleaned.csv`
  - `udemy_cleaned.csv`

Important rules
- Do NOT modify or move any files in the repository's `Data/` directory. The `ml/` code only reads them.
- All model artifacts (e.g. `ml/model.pkl`) are written inside `ml/` and are included in the `.gitignore` added for this folder.

Run example (Windows cmd):

```
set REPO_DATA_DIR="%CD%\..\Data"
python ml\train.py --data-dir "%REPO_DATA_DIR%"
```

Or explicitly:

```
python ml\train.py --data-dir "C:\path\to\repo\Data"
```

If you need training with labeled data, ensure your processed CSVs contain a label column named one of: `target`, `label`, `category`, or `subject`.
