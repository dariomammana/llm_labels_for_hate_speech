# Paper Code (IMSyPP EN/IT/SI)

This repository contains scripts and notebooks for:
- building EN/IT/SI annotation datasets,
- running LLM annotation workflows,
- training/evaluating models,
- comparing agreement and performance metrics.

The repository intentionally does not include raw CSV datasets. All data files are expected to be present locally on the machine running the project.

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

## 2. Local data expectations

This repository assumes the project data is available locally outside GitHub, under folders such as:

- Data/
- Data/Test_data/

The scripts are written to work with local CSV files already present on disk. The repository does not include a data download step and does not track dataset files in Git.

If you want to obtain the source datasets yourself, use the original CLARIN.SI resources here:

- English: https://www.clarin.si/repository/xmlui/handle/11356/1454
- Italian: https://www.clarin.si/repository/xmlui/handle/11356/1450
- Slovenian: https://www.clarin.si/repository/xmlui/handle/11356/1398

The expected local files are:
- Data/IMSyPP_EN_YouTube_comments_train.csv
- Data/IMSyPP_IT_YouTube_comments_train.csv
- Data/IMSyPP_SI_anotacije_round1(in).csv
- Data/Test_data/IMSyPP_EN_YouTube_comments_evaluation_no_context.csv
- Data/Test_data/IMSyPP_IT_YouTube_comments_evaluation.csv
- Data/Test_data/IMSyPP_SI_anotacije_round2.csv

If you have the source data locally, make sure the expected files are available before running the project scripts.

## 3. Main workflows

### 3.1 Annotation batch pipeline

From the annotation directory, per language:

```powershell
python submit_batches_EN.py
python retrieve_results_EN.py
```

Equivalent scripts exist for IT and SI, plus retry/retrieve scripts for SI.

### 3.2 Prompt-design experiments

Run scripts in prompt_design/EN, prompt_design/IT, prompt_design/SI.
Each script writes CSV outputs in its local folder.

### 3.3 Analysis and comparison

Key scripts:
- prompt_design/prompt_comparison.py
- prompt_design/prompt_evaluation.py
- sample_analysis.py
- label_share_summary.py

### 3.4 Model training/evaluation notebooks

Use:
- finetune_xlmr_human.ipynb
- finetune_xlmr_llm.ipynb
- evaluate_models.ipynb
- evaluate_models_bootstrap_holm.ipynb

## 4. Public GitHub safety (no CSV committed)

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

## 5. Notes

- This repo expects local data files under Data/ and Data/Test_data/.
- Some notebooks use Google Drive paths; adjust paths for local-only execution.
- Respect dataset licenses and citation requirements when publishing results.
