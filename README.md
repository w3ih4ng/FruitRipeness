# Fruit Ripeness — Mode A

One continuing VS Code/GitHub project for a five-method image-processing comparison
and a team hybrid. The baseline and all five individual methods (HSV, Otsu,
K-means, GrabCut and marker-controlled watershed) and an HSV + GrabCut union
hybrid are implemented, with a local desktop comparison UI for images and folders.

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

## 3. Dataset and baseline

Your extracted dataset should contain `data/raw/Train` and `data/raw/Test`.
A single wrapper folder such as `data/raw/archive/Train` is accepted too. Keep
source images/labels unchanged. No audit, cleaning or deduplication is required.

After applying the baseline update, run:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
.\.venv\Scripts\python.exe -m fruitripeness.baseline --data data/raw
```

The UI update includes 69 non-GUI tests plus 5 desktop integration tests (skipped
when a graphical Tk display is unavailable). The baseline fits on 3,547 images from original Train and validates
on a fixed 887-image subset; the original 180-image Test set is unchanged and not
scored by default. Results and a model go into a new timestamped folder under
`outputs/baseline/`. The run prints that exact path. See `docs/BASELINE.md`.

The original read-only ZIP audit remains optional. Historical findings are recorded
in `docs/DATASET_AUDIT.md`; they do not trigger image changes or block training.

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

## Open the comparison UI

From this existing project folder:

```powershell
.\.venv\Scripts\python.exe -m fruitripeness.ui
```

Open **Models & runs**, review the discovered runs and confirm your team's model
files are trusted. Then choose images/a folder and run a method or Compare all six.
**Saved validation** loads the measured metrics and confusion matrices, including
the baseline reference. PNG/CSV exports are available. See `docs/UI.md`.

No new pip dependencies, browser server or retraining are required. Tkinter must
be present in your Python installation; test it with `python -m tkinter` if needed.
The original Test set stays reserved; final Test evaluation is a later milestone.

The methods were tested on Python 3.12.13/Linux. The user reproduced baseline, HSV,
Otsu, K-means, GrabCut, Watershed and hybrid results on Windows. The UI core tests
pass on Linux; desktop rendering could not be verified in the headless build
environment. Run the desktop integration tests and inspect the window on Windows.

## Sources

- Dataset: https://www.kaggle.com/datasets/asadullahprl/fruits-ripeness-classification-dataset
- GitHub setup: https://docs.github.com/en/migrations/importing-source-code/using-the-command-line-to-import-source-code/adding-locally-hosted-code-to-github
- VS Code environments: https://code.visualstudio.com/docs/python/environments

The assignment specification and template remain authoritative. Keep an AI usage
record and ensure each member understands and can explain their own contribution.
