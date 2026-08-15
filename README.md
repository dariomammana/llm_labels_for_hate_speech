# Paper Code (IMSyPP EN/IT/SI)

This repository contains scripts and notebooks for:
- building EN/IT/SI annotation datasets,
- running LLM annotation workflows,
- training/evaluating models,
- comparing agreement and performance metrics.

The repository is configured so CSV data files are not committed to GitHub.

## 1. Environment setup

From the repository root:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install pandas numpy scikit-learn torch transformers krippendorff matplotlib seaborn statsmodels anthropic
```

If you run scripts in the annotation folder that call Anthropic, set your API key:

```powershell
$env:ANTHROPIC_API_KEY="your_key_here"
```

## 2. Download source datasets from CLARIN

Run:

```powershell
python download_clarin_data.py
```

This downloads the required files from these handles:
- https://www.clarin.si/repository/xmlui/handle/11356/1454 (EN)
- https://www.clarin.si/repository/xmlui/handle/11356/1450 (IT)
- https://www.clarin.si/repository/xmlui/handle/11356/1398 (SI)

It writes the files to paths expected by the existing scripts, including SI filename normalization:
- Data/IMSyPP_SI_anotacije_round1(in).csv
- Data/Test_data/IMSyPP_SI_anotacije_round2.csv

To force re-download:

```powershell
python download_clarin_data.py --force
```

## 3. Build derived datasets used by this project

```powershell
python human_labels_import.py
python unique_add_index.py
python test_set_cleanup.py
```

These commands generate files like:
- Data/Unique_EN.csv, Data/Unique_IT.csv, Data/Unique_SI.csv
- Data/Sample150_EN.csv, Data/Sample150_IT.csv, Data/Sample150_SI.csv
- Data/EN_human_annotations.csv, Data/IT_human_annotations.csv, Data/SI_human_annotations.csv
- Data/Test_data/*_mapped.csv

## 4. Main workflows

### 4.1 Annotation batch pipeline

From the annotation directory, per language:

```powershell
python submit_batches_EN.py
python retrieve_results_EN.py
```

Equivalent scripts exist for IT and SI, plus retry/retrieve retry scripts for SI.

### 4.2 Prompt-design experiments

Run scripts in prompt_design/EN, prompt_design/IT, prompt_design/SI.
Each script writes CSV outputs in its local folder.

### 4.3 Analysis and comparison

Key scripts:
- prompt_design/prompt_comparison.py
- prompt_design/prompt_evaluation.py
- sample_analysis.py
- label_share_summary.py

### 4.4 Model training/evaluation notebooks

Use:
- finetune_xlmr_human.ipynb
- finetune_xlmr_llm.ipynb
- evaluate_models.ipynb
- evaluate_models_bootstrap_holm.ipynb

## 5. Public GitHub safety (no CSV committed)

The repository .gitignore excludes CSV files globally.

Before pushing, verify no CSV is tracked:

```powershell
git ls-files -- '*.csv'
```

If this ever returns files, untrack them while keeping local copies:

```powershell
git rm --cached path/to/file.csv
git commit -m "Stop tracking dataset CSV"
```

To inspect ignored CSV files:

```powershell
git status --ignored
```

## 6. Notes

- This repo expects local data files under Data/ and Data/Test_data/.
- Some notebooks use Google Drive paths; adjust paths for local-only execution.
- Respect CLARIN dataset licenses and citation requirements when publishing results.
