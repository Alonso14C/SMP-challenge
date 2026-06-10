import os
import pickle
import argparse
import time
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import KFold
from sklearn.metrics import mean_absolute_error
from scipy.stats import spearmanr
from catboost import CatBoostRegressor

def log(msg):
    print(f"[INFO] {msg}", flush=True)


def create_cross_features(df, add_feats):
    cross = pd.DataFrame(
        {
            'views_per_photo': df['totalViews'] / (df['photoCount'] + 1),
            'faves_per_view': df['totalFaves'] / (df['totalViews'] + 1),
            'tags_per_photo': df['totalTags'] / (df['photoCount'] + 1),
            'follower_following_ratio': df['followerCount'] / (df['followingCount'] + 1),
            'social_engagement': df['totalFaves'] / (df['followerCount'] + 1),
            'social_activity': df['totalInGroup'] / (df['photoCount'] + 1),
            'geo_per_photo': df['totalGeotagged'] / (df['photoCount'] + 1),
            'interaction_density': (df['totalTags'] + df['totalInGroup']) / (df['photoCount'] + 1),
        },
        index=df.index,
    )
    cross['content_popularity'] = df['totalViews'] * cross['faves_per_view']
    cross['social_influence'] = df['followerCount'] * cross['faves_per_view']
    cross['growth_potential'] = cross['follower_following_ratio'] * cross['social_engagement']
    return pd.concat([df, cross], axis=1)


def load_pickle(path):
    with open(path, 'rb') as f:
        return pickle.load(f)


def build_data(args):
    data = load_pickle(args.feat_path)
    log(f"base shape={data.shape}")
    data = data.drop(columns=[c for c in args.drop_cols.split(',') if c], errors='ignore')

    log("构造基础交叉特征")
    data = create_cross_features(
        data,
        add_feats=['totalViews', 'totalTags', 'totalGeotagged', 'totalFaves', 'totalInGroup', 'photoCount', 'followerCount', 'followingCount'],
    )

    log(f"合并 w2v: {args.emb_path}")
    w2v = load_pickle(args.emb_path)
    data = data.merge(w2v, on='Uid', how='left')

    train = data[~data['label'].isnull()].reset_index(drop=True)
    test = data[data['label'].isnull()].reset_index(drop=True)
    log(f"train={train.shape} test={test.shape}")

    log(f"读取 train old clip: {args.clip_path}")
    tr_clip = pd.DataFrame(np.asarray(pd.read_pickle(args.clip_path)))
    tr_clip.columns = [f"my1_{i}" for i in range(tr_clip.shape[1])]
    if len(tr_clip) != len(train):
        raise ValueError(f"train clip rows mismatch: clip={len(tr_clip)} train={len(train)}")
    train = pd.concat([train, tr_clip.reset_index(drop=True)], axis=1)

    log(f"读取 test old clip: {args.test_clip_path}")
    te_clip = pd.DataFrame(np.asarray(pd.read_pickle(args.test_clip_path)))
    te_clip.columns = [f"my1_{i}" for i in range(te_clip.shape[1])]
    if len(te_clip) != len(test):
        raise ValueError(f"test clip rows mismatch: clip={len(te_clip)} test={len(test)}")
    test = pd.concat([test, te_clip.reset_index(drop=True)], axis=1)

    return train, test


def process_categorical(train, test, drop_cols):
    candidate_cols = list(dict.fromkeys(
        train.select_dtypes(["object", "category"]).columns.tolist()
        + test.select_dtypes(["object", "category"]).columns.tolist()
        + ['Uid', 'ispro', 'Ispublic', 'datetime_x2', 'hour', 'datetime_x1']
    ))
    cat_cols = []
    for col in candidate_cols:
        if col in drop_cols or col == 'Pid':
            continue
        if col in train.columns and col in test.columns:
            tr_s = train[col].astype('object').where(train[col].notna(), 'Missing').astype(str).replace({'nan': 'Missing', 'None': 'Missing'})
            te_s = test[col].astype('object').where(test[col].notna(), 'Missing').astype(str).replace({'nan': 'Missing', 'None': 'Missing'})
            cats = pd.Index(pd.concat([tr_s, te_s], ignore_index=True).unique())
            train[col] = pd.Categorical(tr_s, categories=cats)
            test[col] = pd.Categorical(te_s, categories=cats)
            cat_cols.append(col)

    label = 'label'
    base_drop = {'label', 'datetime', 'description', 'joinedDate2', 'All_tags_len'} | set(drop_cols)
    feats = [c for c in train.columns if c not in base_drop and c in test.columns]
    cat_cols = [c for c in cat_cols if c in feats and c != 'datetime']
    log(f"feature_count={len(feats)} cat_count={len(cat_cols)}")
    return train, test, feats, cat_cols


def make_model(args):
    return CatBoostRegressor(
        eval_metric='MAE',
        depth=5,
        learning_rate=0.03,
        iterations=5000,
        random_seed=int(args.seed),
        task_type='GPU',   
        devices='0',
        verbose=200
    )


def main():
    parser = argparse.ArgumentParser()
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    parser.add_argument('--feat_path', default=os.path.join(root, 'feat', 'baseFeat.pkl'))
    parser.add_argument('--emb_path', default=os.path.join(root, 'feat', 'w2v_emb.pkl'))
    parser.add_argument('--clip_path', default=os.path.join(root, 'feat', 'metaclip_text.pkl'))
    parser.add_argument('--test_clip_path', default=os.path.join(root, 'feat', 'metaclip_text_test.pkl'))
    parser.add_argument('--drop_cols', default='img_path')
    parser.add_argument('--seed', type=int, default=77)
    parser.add_argument('--n_splits', type=int, default=2)
    parser.add_argument('--lgb_n_estimators', type=int, default=3000)
    parser.add_argument('--lgb_n_jobs', type=int, default=8)
    parser.add_argument('--early_stopping_rounds', type=int, default=200)
    parser.add_argument('--output_dir', default=os.path.join(root, 'subs'))
    parser.add_argument('--model_dir', default=os.path.join(root, 'new_model'))
    parser.add_argument('--tag', default='cat_base_5fold_oldclip_3000')
    parser.add_argument('--clip_min', type=float, default=1.0)
    parser.add_argument('--clip_max', type=float, default=16.56)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(args.model_dir, exist_ok=True)

    train, test = build_data(args)
    drop_cols = [c.strip() for c in args.drop_cols.split(',') if c.strip()]
    train, test, feats, cat_cols = process_categorical(train, test, drop_cols)

    y = train['label'].to_numpy(dtype=np.float32)
    post_ids = test['Pid'].map(lambda x: 'post' + str(x)) if 'Pid' in test.columns else pd.Series([f'post{i}' for i in range(len(test))])
    kf = KFold(n_splits=int(args.n_splits), shuffle=True, random_state=int(args.seed))
    oof = np.zeros(len(train), dtype=np.float32)
    test_preds = []
    fold_rows = []
    start = time.time()

    for fold, (tr_idx, va_idx) in enumerate(kf.split(train), start=1):
        log(f"========== fold {fold}/{args.n_splits} ==========")
        model = make_model(args)
        model.fit(
            train.iloc[tr_idx][feats], y[tr_idx],
            eval_set=[(train.iloc[va_idx][feats], y[va_idx])],
            cat_features=cat_cols,
            early_stopping_rounds=int(args.early_stopping_rounds),
            verbose=False
            
        )
        best_iter = int(getattr(model, 'best_iteration_', None) or args.lgb_n_estimators)
        va_pred = model.predict(train.iloc[va_idx][feats])
        te_pred = model.predict(test[feats])
        oof[va_idx] = va_pred.astype(np.float32)
        test_preds.append(te_pred.astype(np.float32))
        mae = mean_absolute_error(y[va_idx], va_pred)
        sp = spearmanr(y[va_idx], va_pred).correlation
        fold_rows.append({'fold': fold, 'best_iteration': best_iter, 'mae': mae, 'spearman': sp})
        log(f"fold={fold} best_iteration={best_iter} MAE={mae:.6f} Spearman={sp:.6f} elapsed={time.time()-start:.1f}s")
        model_path = os.path.join(args.model_dir, f'{args.tag}_fold{fold}.pkl')
        with open(model_path, 'wb') as f:
            pickle.dump(model, f)
        log(f"saved_model={model_path}")

    oof_mae = mean_absolute_error(y, oof)
    oof_sp = spearmanr(y, oof).correlation
    log(f"oof_metrics: MAE={oof_mae:.6f} Spearman={oof_sp:.6f}")
    fold_df = pd.DataFrame(fold_rows)
    metrics_path = os.path.join(args.model_dir, f'{args.tag}_metrics.csv')
    fold_df.to_csv(metrics_path, index=False)
    log(f"saved_metrics={metrics_path}")

    pred = np.mean(np.vstack(test_preds), axis=0)
    sub = pd.DataFrame({'post_id': post_ids.astype(str), 'popularity_score': pred})
    raw_path = os.path.join(args.output_dir, f'sub_{args.tag}_avg_raw.csv')
    sub.to_csv(raw_path, index=False)
    log(f"saved_raw_submission={raw_path}")

    sub['popularity_score'] = sub['popularity_score'].clip(float(args.clip_min), float(args.clip_max))
    clip_path = os.path.join(args.output_dir, f'sub_{args.tag}_avg_clip1_16p56.csv')
    sub.to_csv(clip_path, index=False)
    s = sub['popularity_score']
    log(f"saved_clipped_submission={clip_path}")
    log(f"submission_stats: n={len(sub)} min={s.min():.6f} max={s.max():.6f} mean={s.mean():.6f} std={s.std():.6f} nan={s.isna().sum()}")


if __name__ == '__main__':
    main()
