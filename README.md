# SMP Challenge - Social Media Popularity Prediction

Solution for SMP Challenge: predict social media image popularity scores using multi-modal data (user profiles, metadata, text tags, spatial-temporal info).

## Structure

```
code/              # Source code
  feature_EDA.py   # Feature engineering & Word2Vec embeddings
  get_feat.py      # Base feature generation
  train_lgb_*.py   # LightGBM training (1-fold / 5-fold)
  train_cat_*.py   # CatBoost training (1-fold / 5-fold)
  fusion.py        # Ensemble: 0.62*LGB + 0.4*CatB
data/              # Raw dataset (train/ & test/)
feat/              # Generated features (.pkl)
embedding/         # Pre-trained Word2Vec models
new_model/         # Saved models
subs/              # Model predictions
submission/        # Final submission CSV
```

## Dependencies

```bash
pip install -r requirement.txt
```

Main libs: `lightgbm==4.2.0`, `catboost==1.2.2`, `pandas`, `numpy`, `scikit-learn`, `gensim`.

## Pipeline

1. **Feature Engineering** — user stats, temporal features, tag parsing, group aggregations, cross features
2. **Embeddings** — Word2Vec (user-tag co-occurrence, dim=32) + MetaCLIP text features
3. **Training** — LightGBM 5-fold CV + CatBoost
4. **Fusion** — weighted ensemble → `submission/sub.csv`

## Quick Start

```bash
bash run.sh
```

Or step by step:
```bash
python code/get_feat.py               # build features
python code/train_lgb_5fold_base.py   # train LightGBM
python code/train_cat_1fold_base.py   # train CatBoost
python code/fusion.py                 # ensemble & submit
```

## Model Config

**LightGBM**
- 5-fold CV, MAE early stopping
- `max_depth=5`, `lr=0.03`, `n_estimators=25000`
- `colsample_bytree=0.2`, `colsample_bynode=0.4`

**CatBoost**
- Native categorical feature support
- Same data pipeline as LightGBM

## Note

Large files (`*.json`, `*.pkl`, `*.pkl` models) are tracked by Git LFS but not pushed to remote due to network limits. Prepare data locally to reproduce.

**Full code archive (with large files):** [Google Drive](https://drive.google.com/file/d/1draBuxT6FrzJaEO0_0idJeh_AtHlq9g5/view?usp=drive_link)
