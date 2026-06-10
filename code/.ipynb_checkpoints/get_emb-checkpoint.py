import pandas as pd
import numpy as np
import pickle
from gensim.models.word2vec import Word2Vec 
import joblib
import itertools
import pdb
import argparse

# class Args:
#     base_feat_path="/root/project/feat/baseFeat.pkl"
#     w2v_load_path="/root/project/w2v_models"
#     emb_save="/root/project/feat/w2v_emb.pkl"
    
# args=Args()

# 创建解析器
parser = argparse.ArgumentParser()

# 添加参数
parser.add_argument('--base_feat_path', type=str, default="/root/project/feat/baseFeat.pkl")
parser.add_argument('--w2v_load_path', type=str,default="/root/project/w2v_models")
parser.add_argument('--emb_save',  type=str,default="/root/project/feat/w2v_emb.pkl" )

# 解析参数
args = parser.parse_args()



with open(args.base_feat_path,"rb") as f:
    data=pickle.load(f)
data=data[['Uid','top1_tags','top2_tags','top3_tags','Alltags','week_top1_tags']].copy()

data.shape

data_temp = data[['Uid']].drop_duplicates()
cnt=1
#flag=1新建模型，为0就是加载
flag=0

def emb(df, f1, f2,cnt,flag,if_list=False):
    emb_size = 32
    tmp = df.groupby(f1, as_index=False)[f2].agg({'{}_{}_list'.format(f1, f2): list})
    if if_list:
        tmp['{}_{}_list'.format(f1, f2)]=tmp['{}_{}_list'.format(f1, f2)].map(lambda x:list(itertools.chain(*x)))
    sentences = tmp['{}_{}_list'.format(f1, f2)].values.tolist()
    print("开始")
    del tmp['{}_{}_list'.format(f1, f2)]
    for i in range(len(sentences)):
        sentences[i] = [str(x) for x in sentences[i]]
    if flag==1:
        model = Word2Vec(sentences, vector_size=emb_size, window=6, min_count=5, sg=1, hs=0, seed=1)
        #保存模型
        joblib.dump(model,f'{args.w2v_load_path}/w2v_model_{cnt}')
    elif flag==0:
        #读取模型
        model=joblib.load(f'{args.w2v_load_path}/w2v_model_{cnt}')
    
    emb_matrix = []
    for seq in sentences:
        vec = []
        for w in seq:
            if w in model.wv:
                vec.append(model.wv[w])
        if len(vec) > 0:
            emb_matrix.append(np.mean(vec, axis=0))
        else:
            emb_matrix.append([0] * emb_size)
    emb_matrix = np.array(emb_matrix)
    for i in range(emb_size):
        tmp['{}_{}_emb_{}'.format(f1, f2, i)] = emb_matrix[:, i]
    cnt=cnt+1
    print("结束")
    return tmp,cnt
# res=[]
# sta,cnt=emb(data,'Uid','week_top1_tags',cnt,flag)
# data_temp=data_temp.merge(sta,on='Uid',how='left')

# sta,cnt=emb(data,'week_top1_tags','Uid',cnt,flag)
# data_temp=data_temp.merge(sta,on='week_top1_tags',how='left')

sta,cnt=emb(data,'Uid','Alltags',cnt,flag)
data_temp=data_temp.merge(sta,on='Uid',how='left')

sta,cnt=emb(data,'Uid','top1_tags',cnt,flag)
data_temp=data_temp.merge(sta,on='Uid',how='left')

sta,cnt=emb(data,'Uid','top2_tags',cnt,flag)
data_temp=data_temp.merge(sta,on='Uid',how='left')

sta,cnt=emb(data,'Uid','top3_tags',cnt,flag)
data_temp=data_temp.merge(sta,on='Uid',how='left')

# sta,cnt=emb(data,'top1_tags','Uid',cnt,flag)
# data_temp=data_temp.merge(sta,on='top1_tags',how='left')

data_temp.to_pickle(args.emb_save)