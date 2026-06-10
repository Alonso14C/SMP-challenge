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

from collections import Counter
import itertools
import time

import argparse

# class Args:
#     a_path="/root/project/data/user_additional_2.csv"
#     train_path="/root/project/data/train/"
#     test_path="/root/project/data/test/"
#     b_path="/root/project/data/user_additional.csv"
#     save_path="/root/project/feat/baseFeat.pkl"

# args=Args()

# 创建解析器
parser = argparse.ArgumentParser()

# 添加参数
parser.add_argument('--a_path', type=str, default="../data/train/user_additional_2.csv")
parser.add_argument('--b_path',  type=str,default="../data/train/user_additional.csv" )
parser.add_argument('--train_path', type=str,default="../data/train/")
parser.add_argument('--test_path',  type=str,default="../data/test/" )
parser.add_argument('--save_path',  type=str,default="../feat/baseFeat.pkl" )

# 解析参数
args = parser.parse_args()

# 忽略特定警告
warnings.filterwarnings("ignore")

print('ok')


add=pd.read_csv(args.a_path)

add['joinedDate']=pd.to_datetime(add['joinedDate'], unit='s')
add=add[['Pathalias','joinedDate']]
add['joinedDate']=add['joinedDate'].astype('int')//10**9

################训练集
path=args.train_path#"/root/project/data/train/"
#text
text=pd.read_json(path+"train_text.json")

# cate
cate= pd.read_json(path+'train_category.json')

#时空
space_time= pd.read_json(path+'train_temporalspatial_information.json')

#用户信息
user_info=pd.read_json(path+'train_user_data.json')

#label
label_data=pd.read_csv(path+"train_label.txt",header=None)
#image
img_data=pd.read_csv(path+"train_img_filepath.txt",header=None)
#add
add_data=pd.read_json(path+"train_additional_information.json")
add_data
for col in ["Pid","Uid"]:
    del cate[col]
    del space_time[col]
    del add_data[col]
img_data.rename(columns={0:"img_path"},inplace=True)



#拼接数据
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
data.to_pickle(args.save_path)
print("Basefeat All right")