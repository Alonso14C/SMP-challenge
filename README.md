# SMP Challenge - Social Media Popularity Prediction

本项目为 **SMP Challenge**（Social Media Prediction Challenge）参赛方案，目标是基于用户画像、图片元数据、时空信息、文本标签等多模态数据，预测社交媒体图片的流行度分数（popularity score）。

## 项目结构

```
SMP-challenge/
├── code/                       # 核心代码
│   ├── feature_EDA.py          # 特征工程 + Word2Vec 嵌入生成
│   ├── get_feat.py             # 基础特征生成（精简版）
│   ├── get_emb.py              # 预训练 Word2Vec 模型加载与嵌入提取
│   ├── train_lgb_5fold_base.py # LightGBM 5折交叉验证训练
│   ├── train_cat_1fold_base.py # CatBoost 单折训练
│   ├── train_cat_5fold_base.py # CatBoost 5折交叉验证训练
│   └── fusion.py               # 多模型结果融合
├── data/                       # 原始数据
│   ├── train/                  # 训练集（JSON / CSV / TXT）
│   └── test/                   # 测试集
├── feat/                       # 生成的特征文件（.pkl）
│   ├── baseFeat.pkl            # 基础特征矩阵
│   ├── w2v_emb.pkl             # Word2Vec 嵌入特征
│   ├── metaclip_text.pkl       # MetaCLIP 文本训练特征
│   └── metaclip_text_test.pkl  # MetaCLIP 文本测试特征
├── embedding/                  # 预训练 Word2Vec 模型
│   └── w2v_model_1~4
├── new_model/                  # 保存的训练好的模型（.pkl）
├── subs/                       # 各模型预测结果
├── submission/                 # 最终提交文件
│   └── sub.csv
├── run.sh                      # 一键运行脚本
└── requirement.txt             # Python 依赖
```

## 环境依赖

```bash
pip install -r requirement.txt
```

核心依赖：
- `lightgbm==4.2.0`
- `catboost==1.2.2`
- `matplotlib==3.8.4`
- `tqdm==4.66.1`

> 注意：实际运行还需 `pandas`, `numpy`, `scikit-learn`, `gensim`, `joblib`, `scipy` 等常用库。

## 数据说明

训练集与测试集包含以下多模态信息：

| 数据文件 | 说明 |
|---------|------|
| `*_user_data.json` | 用户基本信息（粉丝数、照片数、是否 pro 等） |
| `*_text.json` | 图片文本信息（标题、标签、媒体类型） |
| `*_category.json` | 类别信息 |
| `*_temporalspatial_information.json` | 时空信息（发布时间、经纬度等） |
| `*_additional_information.json` | 附加信息（描述、地理位置描述等） |
| `*_img_filepath.txt` | 图片路径 |
| `train_label.txt` | 训练标签（流行度分数） |
| `user_additional.csv` / `user_additional_2.csv` | 补充用户统计信息 |

## 特征工程

### 1. 基础特征
- **用户画像**：`totalViews`, `totalFaves`, `photoCount`, `totalTags`, `followerCount`, `followingCount`, `totalInGroup` 等
- **时间分解**：将 `Postdate` 拆解为年、月、日、时、分、秒、周几、一年中的周数、一月中的第几周
- **照片拍摄时间**：`photo_firstdatetaken` 拆解为年月日时分秒
- **文本长度**：`user_description`, `Title`, `location_description`, `Alltags` 的长度统计
- **标签解析**：从 `Alltags` 提取 `top1_tags` ~ `top5_tags`

### 2. 业务统计特征
- 用户发帖时间间隔统计（min / max / mean / std / skew）
- 每周/每月热门 tags 及其出现次数
- 用户在每周/每月内的发帖数（`Uid_newPid_week_count`, `Uid_newPid_month_count`）
- 用户平均每年发送照片数量
- 注册天数与照片数的交叉特征（`joined_year_view`）

### 3. 分组聚合特征
按 `Uid`, `top1_tags`, `top2_tags`, `top3_tags` 等分组，对数值特征计算：
- 统计量：`max`, `min`, `std`, `skew`, `var`
- 类别计数：`nunique`, `count`, `count/nunique`

### 4. 交叉特征
- `views_per_photo`, `faves_per_view`, `tags_per_photo`
- `follower_following_ratio`, `social_engagement`, `social_activity`
- 数值特征间的加减乘除组合

### 5. Word2Vec 嵌入
- 使用 `gensim.Word2Vec` 训练用户-标签的共现嵌入
- `vector_size=32`, `window=6`, `sg=1` (Skip-gram)
- 生成用户级别的标签嵌入均值向量

### 6. MetaCLIP 文本特征
- 利用 MetaCLIP 模型提取图片标题/描述的语义嵌入
- 作为高维稠密特征输入模型

## 模型方案

### LightGBM（5-Fold）
```python
LGBMRegressor(
    objective='regression',
    metric='mae',
    max_depth=5,
    learning_rate=0.03,
    n_estimators=25000,
    colsample_bytree=0.2,
    colsample_bynode=0.4,
    extra_trees=True,
    n_jobs=20,
    random_state=7,
)
```
- 5 折交叉验证，早停轮数 `early_stopping_rounds=200`
- 评估指标：MAE、Spearman 相关系数
- 预测结果经过 clip 处理（`clip_min=1.0`, `clip_max=16.56`）

### CatBoost
- 支持类别特征自动处理
- 与 LightGBM 类似的数据 Pipeline

### 模型融合
```python
sub['popularity_score'] = res_cat * 0.4 + res_lgb * 0.62
```
- LightGBM 权重 0.62，CatBoost 权重 0.4
- 输出最终提交文件 `submission/sub.csv`

## 快速开始

### 方式一：一键运行
```bash
bash run.sh
```

### 方式二：分步执行

**Step 1：生成基础特征**
```bash
python code/get_feat.py
```

**Step 2：生成 Word2Vec 嵌入（可选，已有预训练模型）**
```bash
python code/feature_EDA.py
```

**Step 3：训练 LightGBM**
```bash
python code/train_lgb_5fold_base.py
```

**Step 4：训练 CatBoost**
```bash
python code/train_cat_1fold_base.py
```

**Step 5：模型融合并生成提交文件**
```bash
python code/fusion.py
```

## 关键参数说明

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--feat_path` | `feat/baseFeat.pkl` | 基础特征文件路径 |
| `--emb_path` | `feat/w2v_emb.pkl` | Word2Vec 嵌入路径 |
| `--clip_path` | `feat/metaclip_text.pkl` | MetaCLIP 训练特征路径 |
| `--test_clip_path` | `feat/metaclip_text_test.pkl` | MetaCLIP 测试特征路径 |
| `--n_splits` | `5` | 交叉验证折数 |
| `--lgb_n_estimators` | `25000` | LightGBM 迭代次数 |
| `--early_stopping_rounds` | `200` | 早停轮数 |
| `--clip_min` / `--clip_max` | `1.0` / `16.56` | 预测值截断范围 |

## 提交格式

最终提交文件 `submission/sub.csv` 格式：

```csv
post_id,popularity_score
post12345,8.92
post12346,12.34
...
```

## 注意事项

1. **大文件**：`data/*.json`、`feat/*.pkl`、`new_model/*.pkl` 等文件体积较大（>100MB），已通过 Git LFS 管理，但因网络限制未推送至远程。如需复现，请在本地准备数据后运行代码生成。
2. **路径问题**：所有脚本使用相对路径，默认以项目根目录为基准运行。
3. **内存需求**：特征工程阶段会产生高维稀疏特征，建议至少 16GB 内存。

## 作者

- GitHub: [@Alonso14C](https://github.com/Alonso14C)
