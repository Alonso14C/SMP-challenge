import pandas as pd
import numpy as np
from catboost import CatBoostRegressor,Pool
import lightgbm as lgb
from sklearn.preprocessing import LabelEncoder
from tqdm.auto import tqdm
from scipy.stats import spearmanr
from sklearn.metrics import mean_absolute_error
import pickle
import warnings
import logging
import argparse
import pdb
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor

import xgboost as xgb
import argparse


# 创建解析器
parser = argparse.ArgumentParser()

# 添加参数
parser.add_argument('--feat_path', type=str, default="/root/project/feat/baseFeat.pkl")
parser.add_argument('--emb_path', type=str,default="/root/project/w2v_models")
parser.add_argument('--clip_path',  type=str,default="/root/project/feat/w2v_emb.pkl" )
parser.add_argument('--model_name', type=str, default="lgb")
parser.add_argument('--emb_path', type=str,default="/root/project/w2v_models")
parser.add_argument('--save_path',  type=str,default="/root/project/models/" )

# 解析参数
args = parser.parse_args()

#读取特征
with open(args.feat_path,"rb") as f:
    data=pickle.load(f)
data.shape

import pandas as pd

def create_cross_features(df, add_feats):
    """
    基于给定的用户特征构造交叉特征
    
    参数:
    df -- 包含原始特征的DataFrame
    add_feats -- 原始特征列表
    
    返回:
    包含原始特征和新构造交叉特征的DataFrame
    """
    # 确保所有需要的原始特征都存在
    for feat in add_feats:
        if feat not in df.columns:
            raise ValueError(f"特征 '{feat}' 不在DataFrame中")
    
    # 创建新特征
    # 1. 互动比率特征
    df['views_per_photo'] = df['totalViews'] / (df['photoCount'] + 1)  # 每张照片的平均浏览量
    df['faves_per_view'] = df['totalFaves'] / (df['totalViews'] + 1)   # 浏览到点赞的转化率
    df['tags_per_photo'] = df['totalTags'] / (df['photoCount'] + 1)    # 每张照片的平均标签数
    
    # 2. 社交网络特征
    df['follower_following_ratio'] = df['followerCount'] / (df['followingCount'] + 1)  # 粉丝关注比
    df['social_engagement'] = df['totalFaves'] / (df['followerCount'] + 1)  # 粉丝参与度(每个粉丝的平均点赞)
    df['social_activity'] = df['totalInGroup'] / (df['photoCount'] + 1)     # 群组活动参与度
    
    # 3. 地理特征
    df['geo_per_photo'] = df['totalGeotagged'] / (df['photoCount'] + 1)     # 地理标记比例
    
    # 4. 复合特征
    df['content_popularity'] = df['totalViews'] * df['faves_per_view']      # 内容受欢迎程度
    df['social_influence'] = df['followerCount'] * df['faves_per_view']     # 社交影响力
    
    # 5. 互动密度特征
    df['interaction_density'] = (df['totalTags'] + df['totalInGroup']) / (df['photoCount'] + 1)
    
    # 6. 增长潜力特征
    df['growth_potential'] = df['follower_following_ratio'] * df['social_engagement']
    
    return df

#使用示例
data = create_cross_features(data, add_feats=['totalViews', 'totalTags', 'totalGeotagged', 'totalFaves',
       'totalInGroup', 'photoCount', 'followerCount', 'followingCount'])


#获取emb信息
with open(args.emb_path,"rb") as f:
    w2v_emb=pickle.load(f)
#合并
data=data.merge(w2v_emb,on='Uid',how='left')

text1=pd.read_pickle(args.clip_path)
text1=pd.DataFrame(np.array(text1))
for col in text1.columns:
     text1.rename(columns={col:f"my1_{col}"},inplace=True)
data=pd.concat([data,text1],axis=1)

# path="../data"
# text1=pd.read_pickle(path+"metaclip_text_test.pkl")
# text1=pd.DataFrame(np.array(text1))
# for col in text1.columns:
#      text1.rename(columns={col:f"my1_{col}"},inplace=True)
# test=pd.concat([test,text1],axis=1)


#划分训练集，验证集
train=data[~data['label'].isnull()].reset_index(drop=True)

#处理字符串特征
lb=LabelEncoder()
cat_cols=[]
for col in tqdm(train.select_dtypes(["object",'category']).columns.tolist()+['Uid','ispro',
                                                                          'Ispublic','datetime_x2',
                                                                          'hour','datetime_x1']):
    if col!='Pid':
        train[col]=train[col].astype(str)
        train[col]=train[col].fillna("Missing")
        train[col]=train[col].astype('category')

        cat_cols.append(col)


label='label'
feats=[col for col in train.columns if col not in [label,'datetime','description','joinedDate2','All_tags_len']]
cat_cols=[a for a in cat_cols if a not in ['datetime']]

print(f"length of feats:{len(feats)}")

#######单折###############
tra=train[:244490].reset_index(drop=True)
vail=train[244490:].reset_index(drop=True)

from sklearn.metrics import mean_squared_error
def custom_mse_loss(y_true, y_pred):
    """
    自定义MSE损失函数（实际与内置MSE相同，仅作示例）
    """
    grad = y_pred - y_true  # 一阶导数（梯度）
    hess = np.ones_like(y_true)  # 二阶导数（Hessian），对MSE恒为1
    return grad, hess

def custom_mse_eval(y_true, y_pred):
    """
    自定义评估函数（用于验证集）
    """
    y_pred = y_pred.reshape(-1, 1)
    loss = mean_squared_error(y_true, y_pred)
    return 'custom_mse', loss, False  # 返回（名称，值，是否越大越好）

def mse_corr_loss(y_true, y_pred):
    # 计算 MSE 部分
    grad_mse = 2 * (y_pred - y_true)  # MSE 的梯度
    hess_mse = 2 * np.ones_like(y_true)  # MSE 的 Hessian

    # 计算 Pearson 相关系数
    y_pred_mean = np.mean(y_pred)
    y_true_mean = np.mean(y_true)
    y_pred_centered = y_pred - y_pred_mean
    y_true_centered = y_true - y_true_mean
    cov = np.mean(y_pred_centered * y_true_centered)
    std_pred = np.std(y_pred)
    std_true = np.std(y_true)
    corr = cov / (std_pred * std_true + 1e-8)

    # 计算相关性部分的梯度（近似）
    grad_corr = -np.tanh((y_true_centered / (std_true * std_pred + 1e-8)))  # 近似梯度
    hess_corr = np.zeros_like(y_true)  # 近似 Hessian（通常设为 0）

    # 组合梯度（alpha=1, beta=1）
    beta = 0.1  # 降低相关性惩罚的权重
    grad = grad_mse + grad_corr
    hess = hess_mse + hess_corr

    return grad, hess
###model define###
def get_model(model_name):
    if model_name=='lgb':

        lgb_params = {
    
            #"objective": "binary",
            "metric": "mae",
            #"objective": mse_corr_loss,
            "metric": "None",
            "max_depth": 10,  
            "learning_rate": 0.05,
            "n_estimators": 2000,  
            "colsample_bytree": 0.8,
            "colsample_bynode": 0.8,
            "verbose": -1,
            "random_state": 42,
            "reg_alpha": 0.1,
            "reg_lambda": 10,
            "extra_trees":True,
            'num_leaves':64,
            'categorical_feature ': 'auto',
            "device": "gpu", 
            "verbose": -1,
            "gpu_use_dp":True,

        }
        model=lgb.LGBMRegressor(**lgb_params)
        return model,model_name
    if model_name=='cat':
        model=CatBoostRegressor(iterations=15000, 
                           eval_metric='MAE',
                           learning_rate=0.03,
                           random_seed=42,
                           logging_level='Verbose',
                           task_type='GPU',#选择GPU模式
                           devices='0',
                            gpu_ram_part=0.6
                        )
        return model,model_name
    if model_name == 'lr':
        
        model = LinearRegression()
        return model, model_name
    
    if model_name == 'ridge':

        model = Ridge(alpha=1.0)
        return model, model_name
    
    if model_name == 'xgb':

        model = xgb.XGBRegressor(
            objective='reg:squarederror',
            learning_rate=0.05,
            n_estimators=2000,
            max_depth=10,
            colsample_bytree=0.8,
            random_state=42,
            tree_method='gpu_hist'  # 使用GPU
        )
        return model, model_name
    
    if model_name == 'rf':
        
        model = RandomForestRegressor(
            n_estimators=100,
            max_depth=10,
            random_state=42,n_jobs=8
        )
        return model, model_name
    
    if model_name == 'gbdt':
        
        model = GradientBoostingRegressor(
            n_estimators=2000,
            learning_rate=0.05,
            max_depth=10,
            random_state=42
        )
        return model, model_name



def get_train(model_name,alls=False):
    print(f"Start train  alls :{alls}  model_name:{model_name}")
    alls=alls
    model_name=model_name
    model,name=get_model(model_name)
    if name=='lgb':
        if alls:
            model=model.fit(train[feats],train[label].values)
            with open(args.save_path+'lgb_model_all.pkl', 'wb') as f:
                pickle.dump(model, f)
        else:
            print("eval")
            eval_result={}
            model=model.fit(tra[feats],tra[label].values,eval_set=[(vail[feats],vail[label].values)],
                                   callbacks = [lgb.log_evaluation(200), lgb.early_stopping(200),lgb.record_evaluation(eval_result)],
                           )
            with open(args.save_path+'lgb_model.pkl', 'wb') as f:
                pickle.dump(model, f)
    if name=='cat':
        if alls:
            train_pool = Pool(train[feats], train[label],cat_features=cat_cols)
            model = model.fit(train_pool,verbose=200)
            with open(args.save_path+'cat_model_all.pkl', 'wb') as f:
                pickle.dump(model, f)
        else:
            train_pool = Pool(tra[feats], tra[label],cat_features=cat_cols)
            val_pool = Pool(vail[feats],vail[label],cat_features=cat_cols)
            model = model.fit(train_pool, eval_set=val_pool,verbose=300)
            with open(args.save_path+'cat_model.pkl', 'wb') as f:
                pickle.dump(model, f)
    if model_name == 'xgb':
        if alls:
            # 使用全部数据进行训练，通常用于最终模型训练
            model.fit(train[feats], train[label].values)
            with open(args.save_path+'xgb_model_all.pkl', 'wb') as f:
                pickle.dump(model, f)
        else:
            print("eval")
            model.fit(tra[feats], tra[label].values,
                      eval_set=[(vail[feats], vail[label].values)],
                      eval_metric="mae",
                      callbacks=[xgb.callback.EarlyStopping(rounds=200, verbose=True)], 
                      verbose=False # 设置为False可以抑制每轮的训练信息
                     )
            with open(args.save_path+'xgb_model.pkl', 'wb') as f:
                pickle.dump(model, f)
    print("训练结束")
get_train(args.model_name,alls=True)

