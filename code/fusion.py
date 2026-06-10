import pandas as pd
import pickle
from tqdm.auto import tqdm
import numpy as np
import argparse
import json
import os



res1 = pd.read_csv('subs/sub_cat.csv')['popularity_score']
res2 = pd.read_csv('subs/sub_lgb.csv')['popularity_score']


sub=pd.DataFrame()
sub['post_id']=res1['Pid']

sub['popularity_score']=res1*0.4+res2*0.62
sub.to_csv(f"subs/sub.csv",index=None)
print("predict successful")