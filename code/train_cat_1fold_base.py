import pandas as pd
import numpy as np
from catboost import CatBoostRegressor, Pool
from tqdm.auto import tqdm
import pickle
import argparse
import os

# ================= 参数 =================
parser = argparse.ArgumentParser()

current_dir = os.path.dirname(os.path.abspath(__file__))
project_dir = os.path.dirname(current_dir)

parser.add_argument('--feat_path', type=str, default=os.path.join(project_dir, "feat", "baseFeat.pkl"))
parser.add_argument('--emb_path', type=str, default=os.path.join(project_dir, "feat", "w2v_emb.pkl"))
parser.add_argument('--clip_path', type=str, default=os.path.join(project_dir, "feat", "metaclip_text.pkl"))
parser.add_argument('--save_path', type=str, default=os.path.join(project_dir, "new_model"))

args = parser.parse_args()

# ================= 读取数据 =================
with open(args.feat_path, "rb") as f:
    data = pickle.load(f)

with open(args.emb_path, "rb") as f:
    w2v_emb = pickle.load(f)

text1 = pd.read_pickle(args.clip_path)
text1 = pd.DataFrame(np.array(text1))
text1.columns = [f"my1_{i}" for i in range(text1.shape[1])]

def create_cross_features(df):
    df['views_per_photo'] = df['totalViews'] / (df['photoCount'] + 1)
    df['faves_per_view'] = df['totalFaves'] / (df['totalViews'] + 1)
    df['tags_per_photo'] = df['totalTags'] / (df['photoCount'] + 1)
    df['follower_following_ratio'] = df['followerCount'] / (df['followingCount'] + 1)
    df['social_engagement'] = df['totalFaves'] / (df['followerCount'] + 1)
    df['social_activity'] = df['totalInGroup'] / (df['photoCount'] + 1)
    df['geo_per_photo'] = df['totalGeotagged'] / (df['photoCount'] + 1)
    df['content_popularity'] = df['totalViews'] * df['faves_per_view']
    df['social_influence'] = df['followerCount'] * df['faves_per_view']
    df['interaction_density'] = (df['totalTags'] + df['totalInGroup']) / (df['photoCount'] + 1)
    df['growth_potential'] = df['follower_following_ratio'] * df['social_engagement']
    return df

data = create_cross_features(data)


data = data.merge(w2v_emb, on='Uid', how='left')
data = pd.concat([data, text1], axis=1)

# ================= 训练集 =================
train = data[~data['label'].isnull()].reset_index(drop=True)

# ================= 类别特征 =================
cat_cols = []
for col in tqdm(train.select_dtypes(["object", "category"]).columns.tolist() +
                ['Uid','ispro','Ispublic','datetime_x2','hour','datetime_x1']):
    if col != 'Pid':
        train[col] = train[col].astype(str).fillna("Missing").astype('category')
        cat_cols.append(col)

label = 'label'
feats = [col for col in train.columns if col not in
         [label, 'datetime', 'description', 'joinedDate2', 'All_tags_len']]

# ================= CatBoost =================
def train_catboost():
    print("Start training CatBoost...")

    model = CatBoostRegressor(
        iterations=28000,
        learning_rate=0.03,
        max_depth=5,
        eval_metric='MAE',
        task_type='GPU',
        devices='0',
        random_seed=42,
        logging_level='Verbose',
        bagging_temperature=0.9,
        gpu_ram_part=0.6
    )

    train_pool = Pool(train[feats], train[label], cat_features=cat_cols)

    model.fit(train_pool, verbose=500)

    os.makedirs(args.save_path, exist_ok=True)
    model_path = os.path.join(args.save_path, 'cat_model_new.pkl')

    with open(model_path, 'wb') as f:
        pickle.dump(model, f)

    print("训练完成 ✅")

# ================= 启动 =================
if __name__ == "__main__":
    train_catboost()


# 创建解析器
parser = argparse.ArgumentParser()

# 添加参数
parser.add_argument('--feat_path', type=str, default="feat/baseFeat.pkl")
parser.add_argument('--emb_path', type=str,default="feat/w2v_emb.pkl")
parser.add_argument('--clip_path',  type=str,default="feat/metaclip_text_test.pkl" )

# 解析参数
args = parser.parse_args()


#读取特征
with open(args.feat_path,"rb") as f:
    data=pickle.load(f)

def create_cross_features(df, add_feats):
    """
    基于给定的用户特征构造交叉特征
    """
    # 检查特征
    for feat in add_feats:
        if feat not in df.columns:
            raise ValueError(f"特征 '{feat}' 不在DataFrame中")
    df['views_per_photo'] = df['totalViews'] / (df['photoCount'] + 1)
    df['faves_per_view'] = df['totalFaves'] / (df['totalViews'] + 1)
    df['tags_per_photo'] = df['totalTags'] / (df['photoCount'] + 1)
    df['follower_following_ratio'] = df['followerCount'] / (df['followingCount'] + 1)
    df['social_engagement'] = df['totalFaves'] / (df['followerCount'] + 1)
    df['social_activity'] = df['totalInGroup'] / (df['photoCount'] + 1)
    df['geo_per_photo'] = df['totalGeotagged'] / (df['photoCount'] + 1)
    df['content_popularity'] = df['totalViews'] * df['faves_per_view']
    df['social_influence'] = df['followerCount'] * df['faves_per_view']
    df['interaction_density'] = (df['totalTags'] + df['totalInGroup']) / (df['photoCount'] + 1)
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



model_path=r"c:\Users\win\Desktop\project\new_model\cat_model_new.pkl"
with open(model_path,"rb") as f:
    model=pickle.load(f)
res=model.predict(test[feats])



sub=pd.DataFrame()
sub['post_id']=test['Pid']
sub['post_id']=sub['post_id'].map(lambda x:"post"+str(x))
sub['popularity_score']=res
sub.to_csv(f"subs/sub_cat.csv",index=None)
print("predict successful")