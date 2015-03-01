# -*- coding: utf-8 -*-
"""
Created on Sat Feb 28 13:18:43 2015

@author: LIght918
updated from local pc
"""
import pandas as pd
import numpy as np
from sklearn import metrics

PATH = 'D:\\weibo\\data\\track1\\'
#TRIM_PATH = 'D:\\weibo\\data\\track1\\trim1000\\'
TRIM_PATH = '/home/light/Documents/ML/kdd2013/data/trim1000/'

NROW = 5000
df_rec_columns = ['userid1','itemid','result','timestamp']
df_user_relation_columns = ['userid1','userid2']
df_user_action_columns = ['userid1','userid2','quoteNum','retweetNum','commNum']
df_user_profile_columns = ['userid1','age','gender','tweetNum','tagid']
df_user_words_columns = ['userid1','words']


## this class trims the original dataset, to and increase modeling speed
class trimDataset:
    df_userdf = pd.DataFrame()
    
    def __init__(self):
        print 'load user related information...'        
        self.trimUser()

    def trimUser(self):
## read user_sns file to get user relation
        df_user_relation = pd.read_csv(PATH+'user_sns.txt', names=df_user_relation_columns, sep='\t', nrows = NROW)
## read the user relation df to get a complete list of user picked, get user related info with all the users        
        userlist = list(set(list(df_user_relation['userid1'])+list(df_user_relation['userid2'])))
        self.df_userdf = pd.DataFrame(userlist, columns=['userid1'])
        df_user_relation.to_csv(TRIM_PATH+'user_sns.txt', index=False, sep='\t')
    
    def trimRec(self):
## get rec according to selected user
        df_rec_tmp = pd.read_csv(PATH+'rec_log_train.txt', names=df_rec_columns, sep='\t')
        df_rec = df_rec_tmp.merge(self.df_userdf, on='userid1', how='inner')
        df_rec.to_csv(TRIM_PATH+'rec_log_train.txt', index=False, sep='\t')
    
    def trim_UserAction(self):
## get user action according to selected user
        df_user_action_tmp = pd.read_csv(PATH+'user_action.txt', names=df_user_action_columns, sep='\t')
        df_user_action = df_user_action_tmp.merge(self.df_userdf, on='userid1', how='inner')
        df_user_action.to_csv(TRIM_PATH+'user_action.txt', index=False, sep='\t')        
    
    def trim_UserProfile(self):
## get user profile according to selected user
        df_user_profile_tmp = pd.read_csv(PATH+'user_profile.txt', names=df_user_profile_columns, sep='\t')
        df_user_profile = df_user_profile_tmp.merge(self.df_userdf, on='userid1', how='inner')
        df_user_profile.to_csv(TRIM_PATH+'user_profile.txt', index=False, sep='\t')    

    def trim_UserKeyword(self):
## get user action according to selected user
        df_user_words_tmp = pd.read_csv(PATH+'user_key_word.txt', names=df_user_words_columns, sep='\t')
        df_user_words = df_user_words_tmp.merge(self.df_userdf, on='userid1', how='inner')
        df_user_words.to_csv(TRIM_PATH+'user_key_word.txt', index=False, sep='\t') 
        
    def trim_item(self):
## get user action according to selected user
        df_user_words_tmp = pd.read_csv(PATH+'user_key_word.txt', names=df_user_words_columns, sep='\t')
        df_user_words = df_user_words_tmp.merge(self.df_userdf, on='userid1', how='inner')
        df_user_words.to_csv(TRIM_PATH+'user_key_word.txt', index=False, sep='\t') 

class Feature:
    df_rec = pd.DataFrame()
    df_user_profile = pd.DataFrame()
    df_user_action = pd.DataFrame()
    df_user_words = pd.DataFrame()
    df_user_relation = pd.DataFrame()    
    df_item = pd.DataFrame()
    
    def __init__(self):
        print 'load user related information...'        
        self.loadDF()
        
    def loadDF(self):
        self.df_rec = pd.read_csv(TRIM_PATH+'rec_log_train.txt',  sep='\t')
        self.df_user_profile = pd.read_csv(TRIM_PATH+'user_profile.txt', sep='\t')
        self.df_user_action = pd.read_csv(TRIM_PATH+'user_action.txt', sep='\t')
        self.df_user_words = pd.read_csv(TRIM_PATH+'user_key_word.txt', sep='\t')
        self.df_user_relation = pd.read_csv(TRIM_PATH+'user_sns.txt',  sep='\t')
        self.df_item = pd.read_csv(TRIM_PATH+'item.txt',names=['itemid','category','keyword'], sep='\t')
        
    def process(self):
        self.df_user_profile = self.df_user_profile[(self.df_user_profile['age']>'1900') & (self.df_user_profile['age']<'2015')]
        self.df_user_profile.age = self.df_user_profile.age.astype(int)
        self.df_user_profile.age = 2014 - self.df_user_profile.age

feature = Feature()
feature.process()


class SimplyCF_item:
    df_rec = pd.DataFrame()
    itemdict = dict()    
    user_item_dict = {} 
    
    def __init__(self,df):    
        self.df_rec = df

## get item list        
    def get_itemList(self):
        itemlist_tmp = list(set(self.df_rec.itemid))
        self.itemdict = {item:0 for item in itemlist_tmp}
    
## get user item dictionary
    def user_item_dict(self):
        user_item_dict = {}        
        item_groupby_user = self.df_rec.groupby(by='userid')
    ## get item list groupby users
        user_item_tuple_list = [item for item in item_groupby_user.itemid]
        for user_item_tuple in user_item_tuple_list:
    ## calculate each item vector, assign item vector of if item exists in itemlist to each user
            item_vector = self.itemdict
            for item in user_item_tuple[1]:
                item_vector[item] = item_vector[item] + 1
            user_item_dict[user_item_tuple[0]] = item_vector
        self.user_item_dict = user_item_dict
        
     def item_similarity(self):
         similarity_item_matrix = pd.DataFrame()
         metrics.pairwise.cosine_distances

feature = Feature()
model = SimplyCF_item(feature.df_rec)
model.get_itemList()
model.user_item_dict()



'''


feature.df_user_profile[feature.df_user_profile['tweetNum']<2000][['gender','tweetNum']].boxplot(by='gender')

# num of actions 
feature.df_rec.groupby(by='userid').sum()[feature.df_rec.groupby(by='userid').sum()['result']>-400]['result'].hist(bins=50)



'''













