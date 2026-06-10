import pandas as pd
import numpy as np
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.metrics import roc_auc_score,mean_squared_error
import lightgbm as lgb
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from tqdm.auto import tqdm
from scipy.stats import spearmanr
from sklearn.metrics import mean_absolute_error
from datetime import datetime
import re
import warnings
from sklearn.feature_extraction.text import CountVectorizer
from collections import Counter
import itertools
import time

import argparse

# 创建解析器
parser = argparse.ArgumentParser()

# 添加参数
parser.add_argument('--a_path', type=str, default="data/train/user_additional_2.csv")
parser.add_argument('--b_path',  type=str, default="data/train/user_additional.csv")
parser.add_argument('--train_path', type=str, default="data/train/")
parser.add_argument('--test_path',  type=str, default="data/test/")
parser.add_argument('--save_path',  type=str, default="feat/baseFeat.pkl")


args = parser.parse_args()


warnings.filterwarnings("ignore")

print('ok')


add=pd.read_csv(args.a_path)

add['joinedDate']=pd.to_datetime(add['joinedDate'], unit='s')
add=add[['Pathalias','joinedDate']]
add['joinedDate'] = add['joinedDate'].astype('int64') // 10**9

path=args.train_path
#text
text=pd.read_json(path+"train_text.json")

# cate
cate= pd.read_json(path+'train_category.json')

space_time= pd.read_json(path+'train_temporalspatial_information.json')

user_info=pd.read_json(path+'train_user_data.json')
label_data=pd.read_csv(path+"train_label.txt",header=None)
img_data=pd.read_csv(path+"train_img_filepath.txt",header=None)
#add
add_data=pd.read_json(path+"train_additional_information.json")
add_data
for col in ["Pid","Uid"]:
    del cate[col]
    del space_time[col]
    del add_data[col]
img_data.rename(columns={0:"img_path"},inplace=True)


data=pd.concat([user_info,add_data,label_data,cate,space_time,img_data,text[['Alltags','Mediatype','Title']]],axis=1)

# data=pd.concat([user_info,label_data],axis=1)
data.rename(columns={0:"label"},inplace=True)
data.columns

############测试集
path=args.test_path#"/root/project/data/test/"
#text
text=pd.read_json(path+"test_text.json")

# cate
cate= pd.read_json(path+'test_category.json')

#时空
space_time= pd.read_json(path+'test_temporalspatial_information.json')

#用户信息
user_info=pd.read_json(path+'test_user_data.json')

#image
img_data=pd.read_csv(path+"test_img_filepath.txt",header=None)
#add
add_data=pd.read_json(path+"test_additional_information.json")
for col in ["Pid","Uid"]:
    del cate[col]
    del space_time[col]
    del add_data[col]
img_data.rename(columns={0:"img_path"},inplace=True)
#拼接数据
testdata=pd.concat([user_info,add_data,cate,space_time,img_data,text[['Alltags','Mediatype','Title']]],axis=1)

print("ok")
####



def get_len(x):
    if "&" in x:
        return len(x.split())
    else:
        return 1

data=pd.concat([data,testdata],axis=0)
data=data.reset_index(drop=True)
#连接add
data=data.merge(add,on='Pathalias',how='left')
add2=pd.read_csv(args.b_path)
#add2=add2[['Pathalias','totalTags','totalGeotagged']]
#连接add2
data=data.merge(add2,on='Pathalias',how='left')
print("ok")

print(data.shape,text.shape)

data['user_description']=data['user_description'].fillna("Missing")
data['length_des']=data['user_description'].map(lambda x:len(x.split(" ")))
data['Category_len']=data['Category'].map(lambda x:get_len(x))

data['Title']=data['Title'].fillna("Missing")
data['length_Title']=data['Title'].map(lambda x:len(x.split(" ")))

data['location_description']=data['location_description'].fillna("Missing")
data['location_description_Title']=data['location_description'].map(lambda x:len(x.split(",")))

data['All_tags_len']=data['Alltags'].map(lambda x:len(x.split(" ")))
data['All_tags_num']=data['Alltags'].map(lambda x:len(x))

def get_tags_top(x,num):
    x=x.split(" ")
    if len(x)>num:
        return x[num]
    else:
        return x[-1]
data['Alltags']=data['Alltags'].fillna("Missing")
data['top1_tags']=data['Alltags'].map(lambda x:get_tags_top(x,0))
data['top2_tags']=data['Alltags'].map(lambda x:get_tags_top(x,1))
data['top3_tags']=data['Alltags'].map(lambda x:get_tags_top(x,2))
data['top4_tags']=data['Alltags'].map(lambda x:get_tags_top(x,3))
data['top5_tags']=data['Alltags'].map(lambda x:get_tags_top(x,4))

num_cols=data.select_dtypes(["int",'float']).columns

cat_cols=data.select_dtypes(['category','object']).columns


def get_t(timestamp):
    timeArray = time.localtime(timestamp)
    datetime = time.strftime("%Y-%m-%d %H:%M:%S", timeArray)
    return datetime

data['photo_firstdatetaken']=data['photo_firstdatetaken'].fillna('1949-01-01 00:00:00')
data['pho_year']=data['photo_firstdatetaken'].map(lambda x:x.split(" ")[0].split("-")[0]).astype('int')
data['pho_month']=data['photo_firstdatetaken'].map(lambda x:x.split(" ")[0].split("-")[1]).astype('int')
data['pho_day']=data['photo_firstdatetaken'].map(lambda x:x.split(" ")[0].split("-")[2]).astype('int')
data['pho_hour']=data['photo_firstdatetaken'].map(lambda x:x.split(" ")[1].split(":")[0]).astype('int')
data['pho_minute']=data['photo_firstdatetaken'].map(lambda x:x.split(" ")[1].split(":")[1]).astype('int')
data['pho_seconds']=data['photo_firstdatetaken'].map(lambda x:x.split(" ")[1].split(":")[2]).astype('int')

def get_myfeat(df):
    data['Postdate2']=data['Postdate'].map(lambda x:get_t(x))
    df['Postdate2']=pd.to_datetime(df['Postdate2'],format='%Y-%m-%d %H:%M:%S')
    df['year']=df['Postdate2'].dt.year
    df['month']=df['Postdate2'].dt.month
    df['day']=df['Postdate2'].dt.day
    df['hour']=df['Postdate2'].dt.hour
    df['minute']=df['Postdate2'].dt.minute
    df['week']=df['Postdate2'].dt.weekday
    del df['Postdate2']
    return df
data=get_myfeat(data)
def get_time_feat(data_df):
    # Temporal-spatial
    data_df['datetime'] = data_df['Postdate'].apply(lambda x: datetime.fromtimestamp(int(x)))
    data_df['datetime_x1'] = data_df['datetime'].apply(lambda x: x.weekday() * 7 + x.hour)
    data_df['datetime_x2'] = data_df['datetime'].apply(lambda x: x.isocalendar()[1])
    data_df['weekofmonth'] = data_df['datetime'].dt.day.apply(lambda x: (x - 1) // 7 + 1)
    return data_df
data=get_time_feat(data)

data['s']=data['Uid'].map(lambda x:x.split("@")[1][1:]).astype(int)


#########业务特征
data['year_month']=data['year'].astype(str)+"-"+data['month'].astype(str)
data['year_month_week']=data['year'].astype(str)+"-"+data['month'].astype(str)+"-"+data['weekofmonth'].astype(str)

data['hour_minute']=data['hour'].astype(str)+"-"+data['minute'].astype(str)

data['newPid']=data.groupby('Title')['Pid'].transform('first')

#用户发的帖子数
data['Uid_newPid_count']=data.groupby('Uid')['newPid'].transform("nunique")
#用户的发帖时间间隔统计
temp=data[['Uid','newPid','Postdate','year_month','year_month_week','Alltags']].drop_duplicates(['Uid','newPid','Postdate']).sort_values(['Uid','Postdate']).reset_index(drop=True)
temp['Uid_post_diff']=temp.groupby('Uid')['Postdate'].diff()
temp['Uid_post_diff_min']=temp.groupby('Uid')['Uid_post_diff'].transform("min")
temp['Uid_post_diff_max']=temp.groupby('Uid')['Uid_post_diff'].transform("max")
temp['Uid_post_diff_mean']=temp.groupby('Uid')['Uid_post_diff'].transform("mean")
temp['Uid_post_diff_std']=temp.groupby('Uid')['Uid_post_diff'].transform("std")
temp['Uid_post_diff_skew']=temp.groupby('Uid')['Uid_post_diff'].transform("skew")

#统计每周内的热门tags
temp['tags_list']=temp['Alltags'].map(lambda x:x.split(" "))
def get_week_top_tags(x,count,if_num):
    x=x.values.tolist()
    #把二维变成一维
    x=list(itertools.chain(*x))
    if if_num:
        return Counter(x).most_common(count+1)[count][1]
    toptags=Counter(x).most_common(count+1)[count][0]
    return toptags

#####统计每周内的热门话题
s=temp.groupby('year_month_week')['tags_list'].agg(lambda x:get_week_top_tags(x,0,False)).reset_index().rename(columns={"tags_list":"week_top1_tags"})
temp=temp.merge(s,on='year_month_week',how='left')
s=temp.groupby('year_month_week')['tags_list'].agg(lambda x:get_week_top_tags(x,0,True)).reset_index().rename(columns={"tags_list":"week_top1_tags_count"})
temp=temp.merge(s,on='year_month_week',how='left')

s=temp.groupby('year_month_week')['tags_list'].agg(lambda x:get_week_top_tags(x,1,False)).reset_index().rename(columns={"tags_list":"week_top2_tags"})
temp=temp.merge(s,on='year_month_week',how='left')
s=temp.groupby('year_month_week')['tags_list'].agg(lambda x:get_week_top_tags(x,1,True)).reset_index().rename(columns={"tags_list":"week_top2_tags_count"})
temp=temp.merge(s,on='year_month_week',how='left')

#####统计每月内的热门话题
s=temp.groupby('year_month')['tags_list'].agg(lambda x:get_week_top_tags(x,0,False)).reset_index().rename(columns={"tags_list":"month_top1_tags"})
temp=temp.merge(s,on='year_month',how='left')
s=temp.groupby('year_month')['tags_list'].agg(lambda x:get_week_top_tags(x,0,True)).reset_index().rename(columns={"tags_list":"month_top1_tags_count"})
temp=temp.merge(s,on='year_month',how='left')

s=temp.groupby('year_month')['tags_list'].agg(lambda x:get_week_top_tags(x,1,False)).reset_index().rename(columns={"tags_list":"month_top2_tags"})
temp=temp.merge(s,on='year_month',how='left')
s=temp.groupby('year_month')['tags_list'].agg(lambda x:get_week_top_tags(x,1,True)).reset_index().rename(columns={"tags_list":"month_top2_tags_count"})
temp=temp.merge(s,on='year_month',how='left')

###每周内用户的发帖数
temp['Uid_newPid_week_count']=temp.groupby(['year_month_week','Uid'])['newPid'].transform('nunique')


#每月内用户的发帖数
temp['Uid_newPid_month_count']=temp.groupby(['year_month','Uid'])['newPid'].transform('nunique')


for col in ['year_month','year_month_week','Alltags','tags_list',]:
    del temp[col]

data=data.merge(temp,on=['Uid','newPid','Postdate'],how='left')
data['cnt']=1
#统计每个月内用户发送的照片数量
data['year_month_Uid_photo_sum']=data.groupby(['year_month','Uid'])['cnt'].transform('sum')

#用户发照片的第一年与最后一年的差值
data['Uid_year_diff']=data.groupby('Uid')['year'].transform("last")-data.groupby('Uid')['year'].transform("first")
#用户平均每年发送照片数量
data['avg_photo_year']=list(map(lambda x,y:x/y if y!=0 else x,data['photo_count'],data['Uid_year_diff']))#data['photo_count']/data['Uid_year_diff']
del data['cnt']

data['Longitude']=data['Longitude'].replace("",0).astype("float")

def get_nums_tags(x):
     # 将 Series 对象中的每个列表转换为一维 NumPy 数组，并将它们堆叠成一个二维数组
    x_array = np.vstack(x.apply(np.array).values)
    # 使用 flatten 来展平成一维数组
    flattened_array = x_array.flatten()
    # 使用 np.unique 获取唯一元素，并计算它们的数量
    unique_elements = np.unique(flattened_array)
    return len(unique_elements)
def get_nums_tags2(x):
    
    # 将分组内的列表转换为字符串，然后计算总长度
    total_length = sum(len(tag) for tag in x)
    return total_length

data['Alltags_split']=data['Alltags'].map(lambda x:x.split())
#data['totalnuniquetags']=data.groupby('Uid')['Alltags_split'].transform(lambda x:get_nums_tags(x))
data['totaltags']=data.groupby('Uid')['Alltags_split'].transform(lambda x:get_nums_tags2(x))
del data['Alltags_split']

data['date_cha']=data['Postdate']-data['joinedDate']
data['joinedDate2']=pd.to_datetime(data['joinedDate'],unit='s')
data['joined_year']=data['joinedDate2'].dt.year

data['joined_year_view']=list(map(lambda x,y,z:x*365*y*100 if z>=1 else x*365*y*10,data['joinedDate'],data['photo_count'],data['ispro']))
data['cnt']=1
data['joined_year2']=data.groupby('Uid')['cnt'].transform('sum')

for col in ['cnt','joinedDate2']:
    del data[col]

# ####一阶
num_cols=['Pid', 'photo_count',  'Postdate','ispro', 'Ispublic','Longitude',
       'Geoaccuracy', 
          'Latitude', 'length_des', 'Category_len', 'length_Title',
       'location_description_Title','All_tags_len','All_tags_num','totaltags','joined_year2','joined_year_view','joinedDate']
#num_cols=[ 'Pid','photo_count', 'Postdate','All_tags_len','length_Title','length_des', 'Category_len','totaltags','joinedDate','totalTags']
s=[]

for col in tqdm(num_cols):
    if col !='label':
        #for col2 in ['year_month','location_description','Pathalias','Uid','Category','Mediatype','s','top1_tags','top2_tags','top3_tags']:
        for col2 in ['Uid','top1_tags','top2_tags','top3_tags',]:
            for meth in ['max','min','std','skew','var']:
                    data[col2+col+meth]=data.groupby(col2)[col].transform(meth)

        if col!='photo_count':
            #data[col+"/photo_count"]=data[col]/data['photo_count']
            data[col+"/photo_count"]=list(map(lambda x,y:x/y if y!=0 else 0,data[col],data['photo_count']))
            s.append(col+"/photo_count")
        #交叉
for col in s:
    for col2 in s:
        if col!=col2:
            data[col+"/"+col2]=data[col]/data[col2]




####次数统计           
for col in tqdm(data.select_dtypes("object").columns.tolist()+['Pid']):
    if col !='Uid':
        for col2 in ['Uid']:
            data[col+col2+"nunique"]=data.groupby(col2)[col].transform("nunique")
            data[col+col2+"count"]=data.groupby([col2,col])[col].transform("count")
            #data[col+col2+"count_rate"]=data[col+col2+"count"]/data.groupby([col2])[col].transform("count")
            data[col+col2+"count/nunique"]=data[col+col2+"count"]/data[col+col2+"nunique"]

col1='Pid'
for col2 in num_cols+s:
    if col1!=col2:
        data[col1+col2+"+"]=data[col1]+data[col2]
        data[col1+col2+"-"]=data[col1]-data[col2]
        data[col1+col2+"*"]=data[col1]*data[col2]
        data[col1+col2+"/"]=data[col1]/data[col2]


# def add_mass_features(df, max_tag_features=500, group_keys=None, numeric_cols=None):
#     """
#     批量增加新特征，目标增加约1000维
#     """
#     # 避免修改原数据
#     df = df.copy()
#     added = 0
    
#     # ==================== 1. 标签词袋特征（约 max_tag_features 维） ====================
#     if 'Alltags' in df.columns and max_tag_features > 0:
#         print("生成标签词袋特征...")
#         # 确保 Alltags 是字符串，缺失填空
#         tags_series = df['Alltags'].fillna('')
#         vec = CountVectorizer(
#             token_pattern=r'(?u)\b\w+\b',
#             max_features=max_tag_features,
#             binary=True          # 只标记出现与否
#         )
#         tag_bow = vec.fit_transform(tags_series)
#         # 转换为 DataFrame 并合并
#         tag_df = pd.DataFrame.sparse.from_spmatrix(
#             tag_bow, 
#             columns=[f'tag_bow_{i}' for i in range(tag_bow.shape[1])]
#         )
#         df = pd.concat([df, tag_df], axis=1)
#         added += tag_bow.shape[1]
#         print(f"  添加了 {tag_bow.shape[1]} 个标签二元特征")
    
#     # ==================== 2. 分组聚合特征 ====================
#     if group_keys is None:
#         # 选择有业务意义的分组键（确保在 df 中）
#         group_keys = ['Uid', 'year_month', 'hour', 'week', 'top1_tags', 'Category']
#         group_keys = [g for g in group_keys if g in df.columns]
    
#     if numeric_cols is None:
#         # 选择关键数值特征（排除 ID 和日期）
#         exclude = ['Pid', 'photo_count', 'Postdate', 'joinedDate', 'label', 'newPid', 'cnt']
#         numeric_cols = [c for c in df.select_dtypes(include=[np.number]).columns 
#                         if c not in exclude and not c.startswith('joined')]
#         # 优先选取重要的
#         priority = ['totalViews', 'totalFaves', 'photoCount', 'totalTags', 'totalGeotagged',
#                     'followerCount', 'followingCount', 'totalInGroup', 'length_Title', 'All_tags_len']
#         numeric_cols = [c for c in priority if c in df.columns] + [c for c in numeric_cols if c not in priority]
#         numeric_cols = numeric_cols[:15]  # 最多15个，控制特征膨胀
    
#     agg_funcs = ['mean', 'std', 'min', 'max', 'median']
#     quantiles = [0.25, 0.75]
#     print("生成分组聚合特征...")
#     for key in group_keys:
#         for col in numeric_cols:
#             # 基本统计量
#             for func in agg_funcs:
#                 new_name = f'{key}_{col}_{func}'
#                 if new_name not in df.columns:
#                     df[new_name] = df.groupby(key)[col].transform(func)
#                     added += 1
#             # 分位数
#             for q in quantiles:
#                 new_name = f'{key}_{col}_q{int(q*100)}'
#                 if new_name not in df.columns:
#                     df[new_name] = df.groupby(key)[col].transform(lambda x: x.quantile(q))
#                     added += 1
#             # 计数
#             new_name = f'{key}_{col}_count'
#             if new_name not in df.columns:
#                 df[new_name] = df.groupby(key)[col].transform('count')
#                 added += 1
#         # 每个分组内的照片数量
#         new_name = f'{key}_photo_count'
#         if 'Pid' in df.columns and new_name not in df.columns:
#             df[new_name] = df.groupby(key)['Pid'].transform('nunique')
#             added += 1
#     print(f"  添加了 {added - (added_before:=added - (added - added_before) if 'added_before' in dir() else 0)} 个聚合特征")
    
#     # ==================== 3. 特征交互与变换 ====================
#     print("生成交互与变换特征...")
#     # 乘积、除法、平方根、对数
#     top_n = min(8, len(numeric_cols))
#     for i in range(top_n):
#         col1 = numeric_cols[i]
#         # 对数变换
#         if f'log1p_{col1}' not in df.columns:
#             df[f'log1p_{col1}'] = np.log1p(df[col1])
#             added += 1
#         # 平方根
#         if f'sqrt_{col1}' not in df.columns:
#             df[f'sqrt_{col1}'] = np.sqrt(np.abs(df[col1]) + 1e-6)
#             added += 1
#         for j in range(i+1, top_n):
#             col2 = numeric_cols[j]
#             # 乘积
#             prod_name = f'{col1}_x_{col2}'
#             if prod_name not in df.columns:
#                 df[prod_name] = df[col1] * df[col2]
#                 added += 1
#             # 除法（避免除0）
#             div_name = f'{col1}_div_{col2}'
#             if div_name not in df.columns:
#                 df[div_name] = df[col1] / (df[col2] + 1e-8)
#                 added += 1
    
#     # 基于时间差的多尺度特征
#     if 'Postdate' in df.columns and 'joinedDate' in df.columns:
#         df['days_since_join'] = (df['Postdate'] - df['joinedDate']) / (3600*24)
#         df['weeks_since_join'] = df['days_since_join'] / 7
#         df['months_since_join'] = df['days_since_join'] / 30
#         added += 3
    
#     # 全局标签频率
#     for tag_col in ['top1_tags', 'top2_tags']:
#         if tag_col in df.columns:
#             freq_map = df[tag_col].value_counts(normalize=True).to_dict()
#             df[f'{tag_col}_global_freq'] = df[tag_col].map(freq_map).fillna(0)
#             added += 1
    
#     print(f"总计新增特征数量: {added}")
#     return df

# data = add_mass_features(data, max_tag_features=500)       
data.to_pickle(args.save_path)
print("Basefeat All right")