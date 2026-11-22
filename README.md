# RSNA 2026 Knee MRI Abnormality Detection

Multi-label detection of 12 knee MRI findings for the [RSNA 2026 Knee Abnormality Detection](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection) Kaggle competition, built around one hard constraint: **only 58 of 4,407 training studies have expert labels.**

The approach that worked best combines **frozen ImageNet ResNet18 features pooled across MRI slices** with **weak labels extracted from radiology reports**. This repo documents the pipeline, the experiments, and, just as importantly, what did not work.

> **Status:** the submission notebook (`notebooks/03_submission_gold_only.ipynb`) ran cleanly end to end on the public 3-study sample, but the scored run on the hidden test set ended in an error. _[update with the cause and, if you fix and resubmit, the final score]_

## The problem

- **Task:** predict 12 binary findings per knee MRI study: ACL, MCL, medial meniscus, lateral meniscus, medial OA, lateral OA, patellofemoral OA, effusion, synovitis, Baker's cyst, contusion, fracture.
- **Training data:** 4,407 studies. 58 are "gold" (expert-adjudicated labels); the other 4,349 come only with a radiology report (many languages, many formats).
- **Test data:** ~1,300 studies **with no report**, so the final model must be image-only.
- **Rules:** code competition, 9-hour runtime limit, no internet at inference.

## Approach

1. **Series selection.** For each plane (axial, coronal, sagittal), prefer a fluid-sensitive series, otherwise fall back to any series of that plane. A rigid "one slot per sequence" input fails because 80.6% of studies have no standard axial series.
2. **Preprocessing.** Resize each slice to 256x256 and normalize it by its own 1st-99th percentile (scanner intensity varies a lot between series).
3. **Features.** Run a frozen ImageNet ResNet18 on every central slice (middle 60%) of each selected series, then **mean- and max-pool across slices**. Result: 3 planes x 1024 features per study.
4. **Gold head.** One strongly regularized logistic regression per finding (`C=0.005`, balanced class weights) trained on the 58 gold studies.
5. **Weak labels from reports.** Negation-aware keyword rules give each report a three-state label per finding: *stated*, *explicitly ruled out*, or *not mentioned* (excluded). A second head trained on these noisy labels (up to ~4,000 studies per finding) is rank-blended 50/50 with the gold head.

### What the submission notebook does

`notebooks/03_submission_gold_only.ipynb` is the image-only inference pipeline for the code-only scoring run:

- It loads a small private *bundle* (a Kaggle dataset, not in this repo) holding the 58 gold studies' pooled features and labels, the ResNet18 backbone weights (no download is possible offline), and a `config.json` of preprocessing settings.
- It fits the 12 gold-only logistic-regression heads, extracts the same features for each test study, and writes `submission.csv` in the sample submission's row order.
- Assertions check the bundle config, the submission schema and the test metadata, so a mismatch fails loudly instead of silently producing wrong predictions.
- The report-derived weak-label blend (Part 2) is **not** part of the submitted notebook. It needs the features for all 4,407 training studies, and the submission uses only the gold-only head (cross-validated macro AUC ~0.669).

## Results

Evaluated on the 58 gold studies, 5-fold multilabel-stratified cross-validation, macro AUC over the 12 findings (pooled out-of-fold predictions unless noted):

| Approach | Macro AUC |
|---|---|
| Fine-tuned ResNet18 on 3-channel MIP input (per-fold mean) | ~0.62 |
| Report-text classifier (not usable at test time) | ~0.615 |
| Frozen ResNet18 features on MIP input + logistic regression | 0.58 |
| Frozen features, **slice-pooled (max)** + logistic regression | **0.669** |
| Slice-pooled + **weak-label blend** (fixed weight 0.5 for all labels) | **0.711** |

**Read these numbers with care.** With 58 studies, a single per-finding AUC carries roughly +/-0.06 of noise, and design choices (pooling type, regularization) were selected on the same 58 studies, so the figures are optimistic. The consistent pattern across configurations is more trustworthy than any single value.

## What I learned

- **A maximum-intensity projection throws away localized findings.** Moving from one MIP per plane to per-slice features with max-pooling took the frozen-feature baseline from 0.58 to 0.67, with the biggest gains on menisci and OA labels.
- **Reports are a noisy proxy for the gold labels, but still useful.** About half of the gold-positive effusion studies have reports that never mention effusion, and "small effusion" in a report often disagrees with gold. Even so, a head trained only on report-derived labels matched or beat the gold-only head for effusion, MCL, medial OA, contusion and synovitis.
- **Weak labels can hurt.** For lateral meniscus and lateral OA the weak head was clearly worse than gold-only (the lateral OA rule agreed with gold only ~36% of the time), so blending is not free.
- **Evaluation needs care at this scale.** Per-fold AUC (about 12 validation studies) is too noisy to compare models; pooled out-of-fold AUC is steadier.
- **Treat the scoring environment as hostile.** In the saved run of the submission notebook, 24 of its ~25 minutes went on a recursive `rglob` over `/kaggle/input`, which walks every DICOM file. A depth-limited search fixes that, and a smoke test on the public sample catches little else, so the scored run needs its own log check.
- **Treat the three report states differently.** "Not mentioned" is not "absent": silent reports must be excluded, not counted as negatives.

## Repository structure

```
notebooks/
  01_preprocessing_and_baselines.ipynb      DICOM exploration, preprocessing, MIP + ResNet18 and text baselines
  02_slice_features_and_weak_labels.ipynb   slice-pooled features, weak labels from reports, blending
  03_submission_gold_only.ipynb             image-only inference notebook for the code-only scoring run
src/
  extract_features.py                       one-notebook, resumable feature extraction for all training studies
```

## Reproducing

The pipeline was developed on Kaggle Notebooks (GPU T4). Data is not included in this repo.

1. Install dependencies with `pip install -r requirements.txt` (on Kaggle the notebooks install what they need).
2. Run `notebooks/01_preprocessing_and_baselines.ipynb` for the data exploration, the preprocessing and the first baselines.
3. Run `src/extract_features.py` once as its own Kaggle notebook (**Save & Run All**, GPU on). It decodes every study and caches the pooled slice features; save the output `all_slice_feats.npz` as a Kaggle dataset.
4. Attach that dataset and run `notebooks/02_slice_features_and_weak_labels.ipynb` (set `FEATS_PATH` in its setup cell).
5. For a submission, build the private bundle (`gold_training.npz` with `ids`, `features` of shape (58, 1536), `labels` and `label_names`; `resnet18_backbone.pth`, the ImageNet ResNet18 state dict without the final layer; and `config.json`), attach it with the competition data, and run notebook 03 with internet off.

## Limitations and next steps

- 58 labeled studies is too few for reliable model selection; confidence intervals are wide.
- Frozen ImageNet features are not tuned for MRI; a stronger or domain-specific backbone, learned slice attention, or using all slices at higher resolution may help.
- The keyword rules are English-centric and the reports are multilingual; a multilingual extraction step would label more of the 4,349 report-only studies.
- Per-label blend weights were not tuned, to avoid overfitting the 58 gold studies.

## Data and licensing

No competition data, radiology reports, labels, or arrays derived from them are included in this repository. Competition data is subject to the Kaggle competition rules. The code is released under the MIT License.
