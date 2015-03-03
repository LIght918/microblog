# -*- coding: utf-8 -*-
"""
Created on Sat Feb 28 13:18:43 2015

@author: LIght918
updated from local pc
"""
import pandas as pd
import numpy as np
import copy
import math
from sklearn import metrics
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier


PATH = 'D:\\weibo\\data\\track1\\'
TRIM_PATH = 'D:\\weibo\\data\\track1\\trim1000\\'
#TRIM_PATH = '/home/light/Documents/ML/kdd2013/data/trim1000/'

NROW = 1000
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
        self.trimRec()
        self.trim_UserAction()
        self.trim_UserProfile()
        self.trim_UserKeyword()
        self.trim_item()

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




class SimplyCF_Model:
    df_rec = pd.DataFrame()
    itemdict = dict()    
    user_item_dict = {} 
    user_item_matrix = {}
    item_item_matrix = pd.DataFrame()
    user_clicked_item = list()
    test = pd.DataFrame()
    
    def __init__(self,df):    
        self.df_rec = df

## get item list        
    def get_itemList(self):
        itemlist_tmp = list(set(self.df_rec.itemid))
        self.itemdict = {item:0 for item in itemlist_tmp}
    
## get user item dictionary
    def user_item_dict(self):
        user_item_dict = {}     
    ## get item that is clicked by users    
        item_groupby_user_clicked = self.df_rec[self.df_rec['result']==1].groupby(by='userid1')
    ## get item list groupby users
        user_item_tuple_list = [item for item in item_groupby_user_clicked.itemid]
        for user_item_tuple in user_item_tuple_list:
    ## calculate each item vector, assign item vector of if item exists in itemlist to each user
            item_vector = copy.deepcopy(self.itemdict)
            for item in user_item_tuple[1]:
                item_vector[item] = item_vector[item] + 1
            user_item_dict[user_item_tuple[0]] = item_vector
        self.user_item_dict = user_item_dict
        
    def item_similarity(self):
        self.user_item_matrix = pd.DataFrame(self.user_item_dict.values())
        self.user_item_matrix.index = self.user_item_dict.keys()
    ## calculate item pairwise similarity        
                
        self.item_item_matrix = pd.DataFrame(1-metrics.pairwise.pairwise_distances(self.user_item_matrix.T ,metric='cosine'))
        self.item_item_matrix.columns = self.user_item_matrix.columns
        self.item_item_matrix.index = self.user_item_matrix.columns   
        self.item_item_matrix = self.item_item_matrix.fillna(0)

## predict the prob of user clicking an item based on his click history
    def calculate_item_clicked_probability(self, userid, itemid):
        user_item_all = self.user_item_matrix.loc[userid,:]
    ## all the item that user has clicked
        user_clicked_item = list(user_item_all[user_item_all>0].index)
        self.user_clicked_item = user_clicked_item
    
    ## get all items interact with itemid in item_item_matrix, and calculate prob
        all_related_item = self.item_item_matrix.loc[itemid,:]
        all_related_item_positive = all_related_item.loc[self.user_clicked_item]
    ## for each clicked vector get a score
        all_related_item_positive_distance = [score*score for score in all_related_item_positive]
        value = sum(all_related_item_positive)*1.0 / math.sqrt(sum(all_related_item_positive_distance))
        return value
    
    def predict(self):
        self.test = self.df_rec[self.df_rec.userid!=387524]
        self.test['score'] = self.test.apply(lambda x: self.calculate_item_clicked_probability(x['userid'],x['itemid']),axis=1)
        self.test['score_norm'] = (self.test['score'] - self.test['score'].mean()) / (self.test['score'].max() - self.test['score'].min())
        self.test['predict'] = self.test.apply(lambda x: 1 if x['score_norm']>0.2 else -1,axis=1)
        
        print 'precision:', metrics.precision_score(self.test['result'], self.test['predict'], labels=[-1,1], pos_label=1)
        print 'recall:', metrics.recall_score(self.test['result'], self.test['predict'], labels=[-1,1], pos_label=1)

class GBRT_model:
    df_rec = pd.DataFrame()  
    user_item_matrix = {}
    item_item_matrix = pd.DataFrame()
    feature_vector = pd.DataFrame
    
    def __init__(self,df_rec,user_item_matrix,item_item_matrix):    
        self.df_rec = df_rec
        self.user_item_matrix = user_item_matrix
        self.item_item_matrix = item_item_matrix

## the following fuction all add features to GBRT, one is user_item_matrix, other is item_item_matrix, the keys are userid and itemid
    def get_feature(self):
    ## drop timestamp, useless for now            
        self.df_rec = self.df_rec.drop('timestamp',axis=1)        
    ## add user_item_matrix to feature        
        self.user_item_matrix['userid1'] = self.user_item_matrix.index
        df1 = self.df_rec.merge(self.user_item_matrix, on='userid1', how='inner')
        df1 = df1.sort(['userid1','itemid'])
        df1_columns = [str(name)+'_user' for name in df1.columns]
        df1.columns = df1_columns
    ## add item_item_matrix to feature
        self.item_item_matrix['itemid'] = self.item_item_matrix.index
        df2 = self.df_rec.merge(self.item_item_matrix, on='itemid', how='inner')
        df2 = df2.sort(['userid1','itemid'])
        df2 = df2.drop(['userid1','itemid','result'],axis=1)
        df2_columns = [str(name)+'_item' for name in df2.columns]
        df2.columns = df2_columns
        df3 = pd.concat([df1,df2],axis=1)
        self.feature_vector = df3
        
    def train_predit(self):
        
        
       
feature = Feature()
feature.process()

model_cf = SimplyCF_Model(feature.df_rec)
model_cf.get_itemList()
model_cf.user_item_dict()
model_cf.item_similarity()
#model_cf.calculate_item_clicked_probability(214028,1775009)
#model_cf.predict()

model_gbrt = GBRT_model(model_cf.df_rec, model_cf.user_item_matrix, model_cf.item_item_matrix)
feature_vector = model_gbrt.get_feature()

'''
trainlabel = feature_vector['result']
train = feature_vector.drop(['userid1','itemid','result'],axis=1)
clf = GradientBoostingClassifier(n_estimators=10, learning_rate=1,max_depth=4, random_state=0)
clf = clf.fit(train, trainlabel)
ypre =clf.predict(train)

print 'precision:', metrics.precision_score(trainlabel, ypre, labels=[-1,1], pos_label=1)
print 'recall:', metrics.recall_score(trainlabel, ypre, labels=[-1,1], pos_label=1)

## random forest classifier

clf = DecisionTreeClassifier(criterion='entropy', max_depth=8, min_samples_split=10)
clf = clf.fit(train, trainlabel)
ypre =clf.predict(train)
print 'precision:', metrics.precision_score(trainlabel, ypre, labels=[-1,1], pos_label=1)
print 'recall:', metrics.recall_score(trainlabel, ypre, labels=[-1,1], pos_label=1)
'''

## random forest classifier adding balancing positive and negative samples
positive_train = feature_vector[feature_vector['result_user']==1]
negative_train = feature_vector[feature_vector.result_user==-1]
binomial_select = np.random.binomial(1,len(positive_train)*10.0/len(negative_train),size=len(negative_train))
negative_train_selected = negative_train[binomial_select==1]

resample_feature_vector = pd.concat([positive_train,negative_train_selected],axis=0)
resample_trainlabel = resample_feature_vector['result_user']
resample_train = resample_feature_vector.drop(['userid1_user','itemid_user','result_user'],axis=1)
clf = DecisionTreeClassifier(criterion='entropy', max_depth=30, min_samples_split=10)
clf = clf.fit(resample_train, resample_trainlabel)
ypre =clf.predict(resample_train)
print 'precision:', metrics.precision_score(resample_trainlabel, ypre, labels=[-1,1], pos_label=1)
print 'recall:', metrics.recall_score(resample_trainlabel, ypre, labels=[-1,1], pos_label=1)
#u:214028 i:1775009 r:1 t:1318382116
'''

## problems
lots of negative label could be labeled as positive label, lower precision of positive labels


'''













