# Clinical Behavioral Testing (Reimplementation Scaffold)

Reimplementation scaffold for:

“What Do You See in this Patient? Behavioral Testing of Clinical NLP Models” (ClinNLP 2022)

This repo provides minimal, runnable code to fine-tune a text classifier, evaluate it, and run simple behavioral tests (e.g., negation minimal pairs) on clinical-style sentences. You can use the structure to plug in your dataset and extend the test suite.

## Quick Links (fill with your shares)
- Code (local): this folder `clinical-behavioral-testing/`
- Models (local): `outputs/` after training
- Data (local): `data/sample/` (tiny demo dataset created)
- Colab (one-click): `colab/Behavioral_Testing.ipynb`
- Drive/HF links: replace later if you publish artifacts

## Environment
- Python 3.10
- Install: `pip install -r requirements.txt`

## Data Format
JSONL with fields:
```
{"text": "The patient has pneumonia.", "label": 1}
```
Place under `data/`:
```
data/
  train.jsonl
  valid.jsonl
  test.jsonl
```
This scaffold includes tiny sample files under `data/sample/` for smoke testing.

## Using the Official Behavioral Testing Repo (bvanaken)
- Upstream project: `https://github.com/bvanaken/clinical-behavioral-testing`
- This scaffold includes a helper script to fetch upstream tests/resources into `external/` and `behavioral_tests/external/`.

Prepare once:
```
bash scripts/prepare_bvanaken.sh
```
- If the upstream repo provides JSON/CSV test resources, they will be copied under `behavioral_tests/external/` and can be passed to `run_tests.py` via `--tests_file`.
- Training datasets are not bundled by the upstream repo; use your approved clinical dataset (or the sample here) formatted as above.

## Train
```
python src/train.py \
  --model_name_or_path bert-base-uncased \
  --train_file data/sample/train.jsonl \
  --validation_file data/sample/valid.jsonl \
  --output_dir outputs/bert-baseline \
  --num_train_epochs 3 \
  --per_device_train_batch_size 16 \
  --learning_rate 2e-5 \
  --seed 42
```

## Evaluate
```
python src/eval.py \
  --model_dir outputs/bert-baseline \
  --test_file data/sample/test.jsonl \
  --metrics f1,precision,recall,accuracy
```
- Metrics are printed and saved to `outputs/bert-baseline/eval_test.json`.

## Behavioral Tests
Run negation-focused minimal pairs (edit/add your own under `behavioral_tests/tests.jsonl`).
```
python src/behavioral_tests/run_tests.py \
  --model_dir outputs/bert-baseline \
  --tests_file behavioral_tests/tests.jsonl \
  --report_path reports/behavioral_report.json
```
The report includes pairwise consistency and expected flip rates.

## Colab
- Open the Colab notebook: `colab/Behavioral_Testing.ipynb`
- Click Runtime → Run all. It installs deps, creates sample data, trains, evaluates, and runs behavioral tests.

## Reproducibility
- Seed: 42
- Hardware: CPU or GPU (Colab T4/A100 recommended)
- Expected behavior on sample data: near-perfect accuracy and high negation flip consistency.

## Notes
- Replace sample data with your real clinical dataset and expand `behavioral_tests/` to match the paper’s categories (negation, lexical variants, abbreviations, demographic mentions, etc.).
- If you prefer GitHub-first distribution: push this repo publicly and link the repo in your report. Host large model artifacts on Drive or Hugging Face and reference them from the README and Appendix.

## Citation
```
@inproceedings{van-aken-etal-2022,
  title = {What Do You See in this Patient? Behavioral Testing of Clinical NLP Models},
  author = {Betty van Aken and Sebastian Herrmann},
  booktitle = {ClinicalNLP 2022},
  year = {2022}
}
```
