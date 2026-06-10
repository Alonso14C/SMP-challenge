import pandas as pd
import pickle
from tqdm.auto import tqdm
import numpy as np
import argparse

# class Args:
#     feat_path='/root/project/feat/baseFeat.pkl'
#     emb_path="/root/project/feat/w2v_emb.pkl"
#     clip_path="/root/project/metaclip_text_test.pkl"
#     model_name1="cat"
#     model_path1=f"/root/project/models/{model_name1}_model_all.pkl"
#     model_name2="lgb"
#     model_path2=f"/root/project/models/{model_name2}_model_all.pkl"

# args=Args()
# 创建解析器
parser = argparse.ArgumentParser()

# 添加参数
parser.add_argument('--feat_path', type=str, default="../feat/baseFeat.pkl")
parser.add_argument('--emb_path', type=str,default="../feat/w2v_emb.pkl")
parser.add_argument('--clip_path',  type=str,default="../feat/metaclip_text_test.pkl" )
parser.add_argument('--model_name1', type=str, default="cat")
parser.add_argument('--model_path1', type=str,default="../models/cat_model_all.pkl")
parser.add_argument('--model_name2',  type=str,default="lgb" )
parser.add_argument('--model_path2',  type=str,default="../models/lgb_model_all.pkl" )

# 解析参数
args = parser.parse_args()


#读取特征
with open(args.feat_path,"rb") as f:
    data=pickle.load(f)
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

# #获取emb信息
# with open("/root/project/feat/r_feat.pkl","rb") as f:
#     r_emb=pickle.load(f)

# data=data.merge(r_emb,on='Pathalias',how='left')



#划分测试集
test=data[data['label'].isnull()].reset_index(drop=True)



text1=pd.read_pickle(args.clip_path)
text1=pd.DataFrame(np.array(text1))
for col in text1.columns:
     text1.rename(columns={col:f"my1_{col}"},inplace=True)
test=pd.concat([test,text1],axis=1)


#处理字符串特征
cat_cols=[]
for col in tqdm(test.select_dtypes(["object",'category']).columns.tolist()+['Uid','ispro',
                                                                          'Ispublic','datetime_x2',
                                                                          'hour','datetime_x1']):
    if col!='Pid':
        test[col]=test[col].astype(str)
        test[col]=test[col].fillna("Missing")
        test[col]=test[col].astype('category')

        cat_cols.append(col)


label='label'
feats=[col for col in test.columns if col not in [label,'datetime','description','joinedDate2','All_tags_len']]
cat_cols=[a for a in cat_cols if a not in ['datetime']]

print(f"length of feats:{len(feats)}")

from sklearn.metrics import mean_squared_error

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

#加载模型
model_name=args.model_name1
model_path=args.model_path1
with open(model_path,"rb") as f:
    model=pickle.load(f)

#预测
res1=model.predict(test[feats])

#加载模型
model_name=args.model_name2
model_path=args.model_path2
with open(model_path,"rb") as f:
    model=pickle.load(f)

#预测
res2=model.predict(test[feats])

sub=pd.DataFrame()
sub['post_id']=test['Pid']
sub['post_id']=sub['post_id'].map(lambda x:"post"+str(x))
sub['popularity_score']=res1*0.6+res2*0.4
sub.to_csv(f"/root/project/subs/sub.csv",index=None)
print("predict successful")