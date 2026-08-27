# Fruit Ripeness — Mode A

One continuing VS Code/GitHub project for a five-method image-processing comparison
and a team hybrid. This first milestone contains a working, read-only dataset audit,
tests and an experiment/UI plan. It is NOT the completed classifier or UI.

## 1. Create the GitHub repository

Create a **private**, empty repository named `FruitRipeness` on GitHub.
Do not initialise it with a README, .gitignore or licence: this folder already has
the required starter files. Keep source private to the approved group/tutor as
required by the assignment. Do not reuse an unrelated previous project repository.

Extract this starter ONCE. Open the actual `FruitRipeness` folder in VS Code.
Later updates modify individual files in this folder.

## 2. Install Python and the environment

Use Python 3.12 and the VS Code Python extension. In the VS Code PowerShell terminal:

```powershell
py -3.12 --version
git --version
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

Select `.venv\Scripts\python.exe` through **Python: Select Interpreter**.
These commands deliberately do not require activating a PowerShell script or changing
your execution policy. If Python 3.12 is missing, install it first; do not substitute
another version without checking dependency compatibility.

macOS/Linux equivalent: `python3.12 -m venv .venv`, then use `.venv/bin/python`
in place of the Windows executable above.

## 3. Place and audit your dataset

Create `data/raw` and copy your supplied `archive.zip` into it. Keep the ZIP intact.
Do not add dataset images to GitHub.

```powershell
.\.venv\Scripts\python.exe -m fruitripeness.audit --zip data/raw/archive.zip
```

The audit decodes every image, records labels and dimensions, computes exact file
and pixel hashes, and flags perceptually similar pairs. It writes:

| Output | Purpose |
|---|---|
| outputs/audit/summary.json | Counts, dimensions and warnings |
| outputs/audit/manifest.csv | One row per image, original path and normalised labels |
| outputs/audit/duplicate_groups.json | Exact duplicates, cross-split and conflicting-label flags |
| outputs/audit/near_duplicate_candidates.csv | Potential similar images requiring review |

The audit does not extract, delete, relabel or modify your dataset. A nonzero exit
status means invalid images were recorded, not that source files were altered.
The initial audit is documented in `docs/DATASET_AUDIT.md`.

## 4. Push this starter

From the project root, replace YOUR_USERNAME with your GitHub account:

```powershell
git init
git branch -M main
git add .
git status
git commit -m "Add dataset audit and Mode A project foundation"
git remote add origin https://github.com/YOUR_USERNAME/FruitRipeness.git
git push -u origin main
```

Before committing, check that `git status` does NOT list archive.zip, images, .venv,
model binaries, credentials or generated audit outputs. Follow the normal GitHub
sign-in prompt; do not paste access tokens into code or chat. If a command fails,
stop and share its non-sensitive error; do not force-push or recreate the folder.

Invite only approved teammates as collaborators through GitHub repository settings.
On another laptop clone this repository and create a new .venv; do not copy .venv.

## Next milestone

Review dataset issues, then create one shared split and a baseline. See
`docs/EXPERIMENT_AND_UI_PLAN.md` for the six-method UI and experimental plan.
No results or models are supplied at this milestone. The dependency pins were tested
with Python 3.12.13 on Linux; Windows setup must be verified on your machine.

## Sources

- Dataset: https://www.kaggle.com/datasets/asadullahprl/fruits-ripeness-classification-dataset
- GitHub setup: https://docs.github.com/en/migrations/importing-source-code/using-the-command-line-to-import-source-code/adding-locally-hosted-code-to-github
- VS Code environments: https://code.visualstudio.com/docs/python/environments

The assignment specification and template remain authoritative. Keep an AI usage
record and ensure each member understands and can explain their own contribution.
