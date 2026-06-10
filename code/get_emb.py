import pandas as pd
import numpy as np
import pickle
from gensim.models.word2vec import Word2Vec 
import joblib
import itertools
import pdb
import argparse


parser = argparse.ArgumentParser()

import os
current_dir = os.path.dirname(os.path.abspath(__file__))   
project_dir = os.path.dirname(current_dir)                
parser.add_argument('--base_feat_path', type=str, default=os.path.join(project_dir, "feat", "baseFeat.pkl"))
parser.add_argument('--w2v_load_path', type=str, default=os.path.join(project_dir, "ebedding")) 
parser.add_argument('--emb_save', type=str, default=os.path.join(project_dir, "feat", "w2v_emb.pkl"))

args = parser.parse_args()



with open(args.base_feat_path,"rb") as f:
    data=pickle.load(f)
data=data[['Uid','top1_tags','top2_tags','top3_tags','Alltags','week_top1_tags']].copy()

data.shape

data_temp = data[['Uid']].drop_duplicates()
cnt=1
flag=0

def emb(df, f1, f2, cnt, flag, if_list=False, model_dir='./models'):
    """
    flag=1: 训练并保存模型
    flag=0: 加载已有模型
    """
    emb_size = 32
    tmp = df.groupby(f1, as_index=False)[f2].agg({'{}_{}_list'.format(f1, f2): list})
    if if_list:
        tmp['{}_{}_list'.format(f1, f2)] = tmp['{}_{}_list'.format(f1, f2)].map(lambda x: list(itertools.chain(*x)))
    sentences = tmp['{}_{}_list'.format(f1, f2)].values.tolist()
    print("开始处理句子...")
    del tmp['{}_{}_list'.format(f1, f2)]
    for i in range(len(sentences)):
        sentences[i] = [str(x) for x in sentences[i]]

    model_path = f'{model_dir}/w2v_model_{cnt}'

    if flag == 1:
        model = Word2Vec(sentences, vector_size=emb_size, window=6, min_count=5, sg=1, hs=0, seed=1)
        joblib.dump(model, model_path)
        print(f"模型已保存至 {model_path}")
    elif flag == 0:
        model = joblib.load(model_path)
        print(f"已加载模型 {model_path}")
    else:
        raise ValueError("flag 必须是 0（加载）或 1（训练）")
    return model

sta,cnt=emb(data,'Uid','Alltags',cnt,flag)
data_temp=data_temp.merge(sta,on='Uid',how='left')
sta,cnt=emb(data,'Uid','top1_tags',cnt,flag)
data_temp=data_temp.merge(sta,on='Uid',how='left')
sta,cnt=emb(data,'Uid','top2_tags',cnt,flag)
data_temp=data_temp.merge(sta,on='Uid',how='left')
sta,cnt=emb(data,'Uid','top3_tags',cnt,flag)
data_temp=data_temp.merge(sta,on='Uid',how='left')

data_temp.to_pickle(args.emb_save)