#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import sys
import tensorflow as tf
import math
from sklearn.preprocessing import StandardScaler
from random import sample
import datetime
import time
sys.path.append('..')
import utils
import json


# In[42]:


utils.show_memory_usage()


# In[7]:


def query_from_mysql(sql):
    host = '192.168.1.240' # '39.108.7.96'
    port = 4000
    user = 'biuser'  # zhangwenyu
    password = 'biuser@2019'
    database = 'charles'
    utils.print_with_datetime('querying %s'%sql)
    return utils.sql_to_df(sql, host, port, database, user ,password)

def query_from_hive(sql):
    host = '172.21.195.22'
    port = 10000
    db = 'source_logs'
    user = 'zhangwenyu'
    password = 'Zhangwy#123456'
    utils.print_with_datetime('querying %s'%sql)
    return utils.sql_to_df(sql, host=host,port=port, db=db, user=user,password=password, db_type='hive')

def split_df_by_column(df, column, train_size=None, split_point=None):
    points = sorted(df[column].values)
    if split_point is None:
        split_index = math.ceil(len(points) * train_size)
        split_point = points[split_index]
    df_train = df[df[column] < split_point]
    df_test = df[df[column] >= split_point]
    return df_train, df_test

def sparseTensor(indices, values, dense_shape, dtype=np.int8):
    l = np.zeros(dense_shape, dtype=dtype)
    for i, sparse_indice in enumerate(indices):
        if type(values) == int or type(values) == float:
            l[sparse_indice[0]][sparse_indice[1]] = values
        else:
            l[sparse_indice[0]][sparse_indice[1]] = values[i]
    return l

class NHotEncoder(utils.LabelEncoder):
    def __init__(self, table_=None):
        super(NHotEncoder, self).__init__(table_)
        
    def fit(self, x):
        for row in x:
            super(NHotEncoder, self).fit(row)
    
    def fit_transform(self, x):
        self.fit(x)
        return self.transform(x)
    
    def transform(self, labels, values=None):
        batch_size = len(labels)
        sparse_indices = []
        sparse_values = []
        
        for i, row in enumerate(labels):
            indices = super(NHotEncoder, self).transform(row)
            row_indices = [[i, index] for index in indices]
            sparse_indices += row_indices
            if values is not None:
                sparse_values += values[i]

        output_shape = (batch_size, len(self.classes_))
        if values is None:
            sparse_values = 1
        y = sparseTensor(indices=sparse_indices, values=sparse_values, dense_shape=output_shape)
        return y


def split_df_by_column(df, column, train_size=None, split_point=None):
    points = sorted(df[column].values)
    if split_point is None:
        split_index = math.ceil(len(points) * train_size)
        split_point = points[split_index]
    df_train = df[df[column] < split_point]
    df_test = df[df[column] >= split_point]
    return df_train, df_test

def normalize(df, fn, columns, suffix=None, inplace=False, param=None):
    if not inplace:
        df = df.copy()
    for col in columns:
        if suffix is not None:
            col = col + '_' + suffix
        if param is not None and col in param:
#             print(param)
            df[col] = fn(df[col], *param[col])
        else:
            df[col] = fn(df[col])

    if not inplace:
        return df

def z_score(df, df_mean = None, df_std=None):
    if df_mean is None or df_std is None:
        return (df - df.mean()) / df.std()
    else:
        return (df - df_mean) / df_std
    
def min_max(df, df_min=None, df_max=None):
    if df_min is None or df_max is None:
        return (df - df.min()) / (df.max() - df.min())
    else:
        return (df - df_min) / (df_max - df_min)
    
def time2sec(t):
    hms = t.strip().split(":")
    if len(hms) == 3:
        return int(hms[0]) * 3600 + int(hms[1]) * 60 + int(hms[2])
    elif len(hms) == 2:
        return int(hms[0]) * 60 + int(hms[1])
    else:
        return int(hms[0])
    
def get_hot_cols(col, n):
    return [col+'_'+str(i) for i in range(n)]


# In[5]:


# make psudo data
# utvid_indexes = list(range(len(encoder.classes_))) # df_train[y_col].unique()
def watch_window_filter(x, window_size=30, targe_col = 'utvId_past', time_col='event_time_past'):
    # sort by time and get the target data within time window size
    sorted_indices = np.argsort(x[time_col])[-window_size:]
    return np.array(x[targe_col])[sorted_indices]

def get_neg_data(df, uid_utvid, utvid_indexes, neg_times=1):
    n_neg = df.shape[0] * neg_times
#     arr_neg = np.zeros((n_neg, df.shape[1]), dtype=int)
    arr_neg = np.empty_like(df, shape=(n_neg, df.shape[1]))
    neg_item_ids = []
    for i in range(len(df)):
        if i % 50000 == 0:
            utils.print_with_datetime(f'Finished {i} negative data')
        non_positive_items = np.setdiff1d(utvid_indexes, uid_utvid[df.iloc[i]['uid']], assume_unique=True)
        non_positive_item_ids = np.random.choice(non_positive_items, neg_times, replace=False)
        neg_item_ids.extend(non_positive_item_ids)
        off_set = i * neg_times
        for j in range(neg_times):
            o = off_set + j
            for c in range(len(df.columns)):
                arr_neg[o,c] = df.iloc[i,c]
#             df_neg.iloc[o]['utvId_now'] = non_positive_item_ids[j]
    df_neg = pd.DataFrame(arr_neg, columns=df.columns)
    df_neg['utvId_now'] = neg_item_ids
    return df_neg

def get_data(neg_times=1, embedding_space={}, hot_space={}):
    sql = 'select uid, utvId, event_time, areaId, completed, collect_day,duration_seconds from xsyx_report_skuinfo.video_user_clean2'
    df_clean =  query_from_mysql(sql)
#     num_video = 
    df_clean2 = df_clean.copy()
    utils.print_with_datetime('Merging df')
    df_merged = df_clean.merge(df_clean2,on=['uid','areaId'], suffixes=('_now','_past'))
    
    # purchasing history timestamp should smaller than this purchasing timestamp
    df_merged = df_merged[df_merged['event_time_past']<df_merged['event_time_now']]
    utils.print_with_datetime('Grouping df')
    df_pos = df_merged.groupby(['uid','areaId','event_time_now'])                            .agg({ 'utvId_now': lambda col: col.tolist()[0],
                                 'completed_now': lambda col: col.tolist()[0],
                                  'collect_day_now': lambda col: col.tolist()[0],
                                  'utvId_past':lambda col: col.tolist(), 
                                  'event_time_past':lambda col: col.tolist(), 
#                                  'completed_past': lambda col: col.tolist(),
#                                 'collect_day_past': lambda col: col.tolist(),
                                 'duration_seconds_past': lambda col: col.tolist()})\
                            .reset_index()
    
    df_pos['is_weekend'] = df_pos['event_time_now'].apply(lambda x: 1 if datetime.datetime.fromtimestamp(x/1e3).isoweekday()>5 else 0)
#     df_pos.to_csv('df_pos.csv', index=False)
#     df_pos = pd.read_csv('df_pos.csv')
    utils.print_with_datetime('Filtering watch windows')
#     df_pos['utvId_past'] = df_pos.apply(watch_window_filter, axis=1)
    embedding_space = {'tagId':{'embedding_size':4}, 'utvid':{'embedding_size':128}, 'authorId':{'embedding_size':4},
                       'areaId':{'embedding_size':3}} #, 'uid':{'embedding_size':8}
    encoder = utils.LabelEncoder()
    encoder.fit(df_pos['utvId_now'].values)
    embedding_space['utvid']['encoder'] = encoder
#     hot_space['utvid'] = {'encoder': encoder}
#     item_embedding_shape = []
    user_time = {}
    user_time_item = {}
    item = {}
    
    user_time['pk'] = {'uid','collect_day_now','event_time_now'}
    user_time['categorical'] = {'multivalent':{'embedding': [{'space_id':'utvid', 'col':'utvId_past'}]},
                                'univalent':{'embedding':[{'space_id':'areaId', 'col':'areaId'}], 
                                             'raw':['is_weekend']}}
#     user_time['categorical'] = {'multivalent':{'hot': [{'space_id':'utvid', 'col':'utvId_past'}]},
#                                 'univalent':{'hot':[{'space_id':'areaId', 'col':'areaId'}],
#                                                     {'space_id':'uid', 'col':'uid'}], 
#                                              'raw':['is_weekend']}}
    user_time['continuous'] = {'aggregative':['duration_seconds_past']}
#     user_time['data'] = df[['uid','collect_day_now','event_time_now','utvId_past','areaId','duration_seconds_past']]
    
    user_time_item['categorical'] = {'univalent':{'embedding': [{'space_id':'utvid', 'col':'utvId_now'}]}}
    user_time_item['reference'] = {'user_time':{'uid':'uid', 'event_time_now':'event_time_now'},
                                   'item':{'utvId_now':'utvId'}} # this_col:that_col
#     user_time_item['data'] = df[['uid','event_time_now','utvId_now']]
    
    sql_utvid_auth = 'SELECT id, authorId, likeNum, playNum, duration, tmCreate as uploadtime FROM xsyx_frxs_base.t_utv'
    df_utvid_auth = query_from_mysql(sql_utvid_auth)
#     df = df.merge(df_utvid_auth, left_on='utvId_now', right_on='id', how='left')
    sql_utvid_tags = 'SELECT utvId, tagId FROM xsyx_frxs_base.t_utv_tag_rel v'
    df_utvid_tags = query_from_mysql(sql_utvid_tags)
    
    df_utv = df_utvid_auth.merge(df_utvid_tags, left_on='id', right_on='utvId', how='inner')
    del df_utv['id']
    df_utv = df_utv.groupby(['utvId','authorId', 'likeNum', 'playNum','duration','uploadtime']).agg({'tagId': lambda col: col.tolist()}).reset_index()
    df_utv['duration_seconds'] = df_utv['duration'].apply(lambda x : time2sec(x))
    
    item['pk'] = {'utvId'}
    item['categorical'] = {'multivalent':{'embedding': [{'space_id':'tagId', 'col':'tagId'}]},
                           'univalent':{'embedding':[{'space_id':'authorId', 'col':'authorId'}],
                                        'raw': ['uploadtime']}}
#     item['categorical'] = {'multivalent':{'hot': [{'space_id':'tagId', 'col':'tagId'}]},
#                            'univalent':{'hot':[{'space_id':'authorId', 'col':'authorId'}]}}
    item['continuous'] = {'raw':['likeNum', 'playNum', 'duration_seconds']}
#     item['data'] = df_utv
    
    utvid_indexes = df_pos['utvId_now'].unique()
    uid_utvid = {}
    for uid in df_pos['uid'].unique():
        df_uid = df_pos[df_pos['uid'] == uid]
        video_ids_watched = np.concatenate(df_uid['utvId_past'].values)
        video_ids_watched = np.union1d(video_ids_watched, df_uid['utvId_now'].values) # result is already unique sorted 
        uid_utvid[uid] = list(video_ids_watched) # set not work for np.setdiff1d(a,b) where a and b must be list

    df_pos['y'] = 1
    if neg_times > 0:
        df_neg = get_neg_data(df_pos, uid_utvid, utvid_indexes, neg_times=neg_times)
        df_neg['y'] = 0
        print(df_pos.shape, df_neg.shape)
        df = pd.concat([df_pos, df_neg])
        df = df.reset_index() # ValueError: Shape of passed values is (1797396, 20), indices imply (1198264, 20)
    else:
        df = df_pos
        
    nested_data = [{'data':df_pos, 'features': {'user_time':user_time}},
                   {'data':df, 'features': {'user_time_item':user_time_item}},
                   {'data':df_utv, 'features':{'item':item}}]
#     embedding_size = {'tagId':4, 'utvid':128}
    
    utils.print_with_datetime(f'Finished getting data')
    return nested_data, embedding_space, hot_space
    
def process_data(nested_data, split_fn=None, topk_preds=None, need_scale=True,
                 cont_cols = [], scaler=None, embedding_space={}, hot_space={}, 
                 pk={}, reference={}):
    # neg_kv = neg_key: {neg_values: neg_samples}, e.g. uid: {utvId_now:[neg_samples]}
    # hot_space = col: encoder
    # embedding_space = [space_id: {'embedding_size':embedding_size, 'encoder':encoder,
    #                               'univalent':col, 'multivalent':col}]

    # pk = {name: dataframe}
    # reference = {'reference':{this_col:that_col}, 'data':dataframe}
    
    for data in nested_data:
        features = data['features']
        df = data['data']
        
        for name, feats in features.items():
            print(name, df.shape)
            df_list = []
            
            utils.print_with_datetime(f'Processing {name}')
            if 'categorical' in feats:
                categorical_feats = feats['categorical']
                if 'univalent' in categorical_feats:
                    uni_cate_feats = categorical_feats['univalent']
                    if 'hot' in uni_cate_feats:
                        uni_cate_feats_hot = uni_cate_feats['hot']
                        for space in uni_cate_feats_hot:
                            space_id = space['space_id']
                            col = space['col']
                            if space_id not in hot_space:
                                hot_space[space_id] = {}
                            if 'encoder' in hot_space[space_id]:
                                one_hot_encoder = hot_space[space_id]['encoder']
                                if type(one_hot_encoder) == utils.LabelEncoder:
                                    one_hot_encoder = utils.OneHotEncoder(one_hot_encoder.table_)
                            else:
                                one_hot_encoder = utils.OneHotEncoder()
                                one_hot_encoder.fit(df[col].values)
                                hot_space[space_id]['encoder'] = one_hot_encoder
                            one_hot_values = one_hot_encoder.transform(df[col].values)
#                             df = pd.concat([df, pd.DataFrame(one_hot_values)], axis=1)
                            df_tmp = pd.DataFrame(one_hot_values, columns=get_hot_cols(col, len(one_hot_encoder.classes_)))
#                             print(130, df_tmp.shape)
                            df_list.append(df_tmp)

                    if 'embedding' in uni_cate_feats:
                        uni_cate_feats_emb = uni_cate_feats['embedding']
                        for embedding in uni_cate_feats_emb:
                            space_id = embedding['space_id']
                            col = embedding['col']
                            if space_id not in embedding_space:
                                embedding_space[space_id] = {}
                            
                            if 'encoder' in embedding_space[space_id]:
                                encoder = embedding_space[space_id]['encoder']
                            else:
                                encoder = utils.LabelEncoder()
                                encoder.fit(df[col].values)
                                embedding_space[space_id]['encoder'] = encoder
                            
                            df_tmp = pd.DataFrame(encoder.transform(df[col].values), columns=[col+'_emb'])
                            df_list.append(df_tmp)

                        embedding_space[space_id]['univalent'] = col+'_emb'
                        
                    if 'raw' in uni_cate_feats:
                        uni_cate_feats_raw = uni_cate_feats['raw']
                        df_list.append(df[uni_cate_feats_raw])
                
                if 'multivalent' in categorical_feats:
                    multi_cate_feats = categorical_feats['multivalent']
                    if 'hot' in multi_cate_feats:
                        multi_cate_feats_hot = multi_cate_feats['hot']
                        for space in multi_cate_feats_hot:
                            space_id = space['space_id']
                            col = space['col']
                            if space_id not in hot_space:
                                hot_space[space_id] = {}
                            if 'encoder' in hot_space[space_id]:
                                n_hot_encoder = hot_space[space_id]['encoder']
                                if type(n_hot_encoder) == utils.LabelEncoder:
                                    n_hot_encoder = NHotEncoder(n_hot_encoder.table_)
                            else:
                                n_hot_encoder = NHotEncoder()
                                n_hot_encoder.fit(df[col].values)
                                hot_space[space_id]['encoder'] = n_hot_encoder
                            print(col)
                            n_hot_values = n_hot_encoder.transform(df[col].values)
#                             df = pd.concat([df, pd.DataFrame(n_hot_values)], axis=1)
                            print(164, n_hot_values.shape)
                            df_list.append(pd.DataFrame(n_hot_values, columns=get_hot_cols(col, len(n_hot_encoder.classes_))))
                            
                    if 'embedding' in multi_cate_feats:
                        multi_cate_feats_emb = multi_cate_feats['embedding']
                        for embedding in multi_cate_feats_emb:
                            space_id = embedding['space_id']
                            col = embedding['col']
                            if space_id not in embedding_space:
                                embedding_space[space_id] = {}
                            if 'encoder' in embedding_space[space_id]:
                                encoder = embedding_space[space_id]['encoder']
                            else:
                                encoder = utils.LabelEncoder()
                                for i in range(len(df[col])):
                                    encoder.fit(df[col].iloc[i])
                                embedding_space[space_id]['encoder'] = encoder

                            df_tmp = pd.DataFrame(df[col].apply(lambda x : encoder.transform(x)).values, columns=[col+'_emb'])
                            df_list.append(df_tmp)
                            embedding_space[space_id]['multivalent'] = col+'_emb'
                            
            if 'continuous' in feats:
                continuous_feats = feats['continuous']
                if 'aggregative' in continuous_feats:
                    agg_cont_feats = continuous_feats['aggregative']
                    for col in agg_cont_feats:
#                         df[col+'_mean'] = df[col].apply(lambda l : np.mean(l))
#                         df[col+'_min'] = df[col].apply(lambda l : min(l))
#                         df[col+'_max'] = df[col].apply(lambda l : max(l))
                        df_list.append(pd.DataFrame(df[col].apply(lambda l : np.mean(l)).values, columns=[col+'_mean']))
                        df_list.append(pd.DataFrame(df[col].apply(lambda l : min(l)).values, columns=[col+'_min']))
                        df_list.append(pd.DataFrame(df[col].apply(lambda l : max(l)).values, columns=[col+'_max']))
                        cont_cols.extend([col+'_mean', col+'_min', col+'_max'])
                if 'raw' in continuous_feats:
                    raw_cont_feats = continuous_feats['raw']
                    df_list.append(df[raw_cont_feats])
                    cont_cols.extend(raw_cont_feats)
                # norm
            if 'reference' in feats:
                cols = [x for v in feats['reference'].values() for x in v.keys()]
#                 print(feats['reference'].values())
#                 print(cols)
                df_list.append(df[cols])
                reference['data'] = pd.concat(df_list, axis=1)
                if 'y' in df:
                    reference['data']['y'] = df['y']
                reference['reference'] = feats['reference']
            elif 'pk' in feats:
                cols = list(feats['pk'])
#                 print(feats['pk'])
#                 print(cols)
                df_list.append(df[cols])
#                 return df_list
#                 for l in df_list:
#                     print(l.shape)
                pk[name] = pd.concat(df_list, axis=1)
        
    processed_data = reference['data']
    print(processed_data.shape)
    for name, cols in reference['reference'].items():
        print(cols, processed_data.columns)
        print(name, pk[name].shape, pk[name].columns)
        processed_data = processed_data.merge(pk[name], left_on = list(cols.keys()), right_on=list(cols.values()), how='left')
        print(processed_data.shape)
    if split_fn is not None:
        df_train, df_test = split_fn(processed_data)
        
        if len(cont_cols) > 0 and need_scale:
            if scaler is None:
                scaler = StandardScaler()
                scaler.fit(df_train[cont_cols])
            df_train[cont_cols] = scaler.transform(df_train[cont_cols])
            df_test[cont_cols] = scaler.transform(df_test[cont_cols])
            utils.print_with_datetime(f'Finished processing data')
            return df_train, df_test, reference, pk, cont_cols, scaler, embedding_space, hot_space
        return df_train, df_test, reference, pk, cont_cols, embedding_space, hot_space
    
    if len(cont_cols) > 0 and need_scale:
        if scaler is None:
            scaler = StandardScaler()
            scaler.fit(processed_data[cont_cols])
        processed_data[cont_cols] = scaler.transform(processed_data[cont_cols])

    return processed_data, reference, pk, cont_cols, embedding_space, hot_space


# In[8]:


np.random.seed(10)
data, embedding_space, hot_space = get_data(neg_times=4)
df_user_time_item = data[0]['data']
split_fn = lambda df: split_df_by_column(df, column='collect_day_now', split_point='2020-07-29')

df_train, df_test, reference, pk, cont_cols, scaler, embedding_space, hot_space =                                         process_data(data, split_fn=split_fn, embedding_space=embedding_space, hot_space=hot_space)


# In[9]:


def postprocess(df_train, col_fn_map, axis=1, need_scale=True, df_test=None, scaler=None, col_fn_map_test=None):
    for col, fn in col_fn_map.items():
        df_train[col] = df_train.apply(fn, axis=axis)
        if df_test is not None:
            df_test[col] = df_test.apply(fn, axis=axis)


    cols = list(col_fn_map.keys())
    if need_scale:
        if scaler is None:
            scaler = StandardScaler()
            scaler.fit(df_train[cols])
        df_train[cols] = scaler.transform(df_train[cols])
        if df_test is not None:
            df_test[cols] = scaler.transform(df_test[cols])
    
    if df_test is not None:
        return df_train, df_test, scaler
    else:
        return df_train, scaler

event_time_now_max = df_train['event_time_now'].max()
col_fn_map = {'now_uploadtime': lambda x: x['event_time_now']//1e3 - datetime.datetime.strptime(str(x['uploadtime']), '%Y-%m-%d %H:%M:%S').timestamp(),
              'example_age': lambda x: (event_time_now_max - x['event_time_now']) / 1000, 
              'example_age_square': lambda x: x['example_age']**2,
              'example_age_rootsquare': lambda x: x['example_age']**0.5}

df_train, post_scaler = postprocess(df_train=df_train, col_fn_map=col_fn_map)


# In[127]:


df_train.head()


# In[28]:


df_train.shape, df_test.shape


# In[10]:


def residual_block(X):
    out = tf.keras.layers.Dense(X.shape[1], activation=tf.nn.relu)(X)
    out = tf.keras.layers.Dense(X.shape[1])(X)
    return tf.nn.relu(X + out)

def deep_crossing(X, n_res_block=5):
    for i in range(n_res_block):
        X = residual_block(X)
    return X

def wide_and_deep(X, wide_network, deep_network, combination_layer):
    return combination_layer(wide_network(X), deep_network(X))
    
def wide_crossing(X, n_cross=5):
    X_l = X
    for i in range(n_cross):
        W_l = tf.Variable(tf.random.truncated_normal([X.shape[1]], stddev=0.01), name = f'cross_weight_{i}')
        b_l = tf.Variable(tf.zeros_initializer()(shape=[X.shape[1]]), name = f'cross_bias_{i}')
        X_l = X * X_l * W_l + b_l + X_l

    return X_l

def combination_layer(wide_out, deep_out):
    return tf.concat([wide_out, deep_out], axis=1)
    


# In[18]:


m = np.concatenate([np.ones([4,3]),np.zeros([4,2])], axis=-1)
key = np.ones([4,5])
q = np.ones([5])
key * q


# In[11]:


tf.where([True,False,True],[1,2,4],[3,2,5]).shape[0]


# In[30]:


def model_fn_builder(embedding_shapes, num_layers = [2048,1024,512,256], k = 10, init_checkpoint=None, learning_rate=1e-4, 
                     drop_out_rate = 0.2, num_train_steps=None, num_warmup_steps=None, max_seq_length = 50):
    
    def model_fn(features, labels, mode):
#         embedding_shapes = {space_id:embedding_shape}
#         features = {'cont_features':tensor, 
#                     'embedding_features':{space_id: {'fixed':tensor, 'ragged':sparse_tensor}}}
        cont_features = features['cont_features']
        embedding_features = features['embedding_features']
#         print(features)
        top_k = k
        embeddings = []
        attention = {}
        for embeddings_name, embeddings_value in embedding_features.items():
            embedding_shape = embedding_shapes[embeddings_name]['embedding_shape']
            embedding = tf.Variable(
                            tf.random.truncated_normal(embedding_shape, stddev=0.01), # 1.0 / math.sqrt(embedding_shape[1])
                            name = embeddings_name)
            if 'univalent' in embeddings_value:
                embedding_fixed_ids = embeddings_value['univalent']
                embedding_fixed = tf.nn.embedding_lookup(embedding, embedding_fixed_ids)
                bias = tf.Variable(tf.zeros_initializer()(shape=[1]), name=f'{embeddings_name}_univalent_bias')
                embedding_fixed = tf.nn.relu(embedding_fixed + bias)
                embeddings.append(embedding_fixed)
                
            if 'multivalent' in embeddings_value:
                embedding_ragged_ids = embeddings_value['multivalent']
                embedding_ragged = tf.nn.embedding_lookup_sparse(embedding, embedding_ragged_ids, None, combiner="mean")
                bias = tf.Variable(tf.zeros_initializer()(shape=[1]), name=f'{embeddings_name}_multivalent_bias')
                embedding_ragged = tf.nn.relu(embedding_ragged + bias)
                embeddings.append(embedding_ragged)
            
            if 'fixed_id' in embeddings_value:
                embedding_fixed_ids_truncated = embeddings_value['fixed_id']
                embedding_fixed_truncated = tf.nn.embedding_lookup(embedding, embedding_fixed_ids_truncated)
                seq_true_length = embeddings_value['true_length']
                attention['keys'] = embedding_fixed_truncated
                attention['masks'] = tf.sequence_mask(seq_true_length, max_seq_length)
                attention['query'] = embedding_fixed
        print(attention)
        # {'keys': <tf.Tensor 'embedding_lookup_1/Identity_1:0' shape=(None, 50, 128) dtype=float32>, 'masks': <tf.Tensor 'SequenceMask/Less:0' shape=(None, 50) dtype=bool>, 'query': <tf.Tensor 'Relu_1:0' shape=(None, 128) dtype=float32>}
        # Dimensions must be equal, but are 5 and 128 for 'mul' (op: 'Mul') with input shapes: [4,5], [?,128].
        attention_out = attention_module(attention['query'], attention['keys'], attention['masks'])
        print(attention_out)
        features = embeddings + [cont_features]
        input_embedding = tf.concat(features, axis=1)

        print("the shape of input_embedding is:", input_embedding.shape)

#         for i, num_layer in enumerate(num_layers):
#             print(i,num_layer)
#             input_embedding = tf.keras.layers.Dense(num_layer, activation=tf.nn.relu, name=f"layer_{i}")(input_embedding)
#             if mode == tf.estimator.ModeKeys.TRAIN:
#                 input_embedding = tf.nn.dropout(input_embedding, drop_out_rate, name=f"layer_dropout_{i}")
                

#             input_embedding = tf.layers.dense(input_embedding, num_layer, activation=tf.nn.relu,
#                                   kernel_initializer=tf.initializers.random_normal(mean=0.0, stddev=0.1),
#                                   bias_initializer=tf.initializers.random_normal(mean=0.0, stddev=0.1), name=f"layer_{i}")
        #   input_embedding = deep_crossing(input_embedding)
        input_embedding = wide_and_deep(input_embedding, wide_crossing, deep_crossing, combination_layer)
        
        logits = tf.keras.layers.Dense(1, name="layer_output")(input_embedding)
        probabilities = tf.nn.sigmoid(logits)  # num * 1
        predictions = tf.cast(probabilities > 0.5, tf.float32)
        
#         logits = tf.matmul(user_vector, item_embedding, transpose_b=True)
        if mode == tf.estimator.ModeKeys.PREDICT:
            predictions = {
                "probabilities": probabilities,
                "predictions": predictions
            }
            export_outputs = {
                "export_outputs": tf.estimator.export.PredictOutput(predictions) # predicted_labels
            }
            return tf.estimator.EstimatorSpec(mode, predictions=predictions, export_outputs=export_outputs)
        else:

#             one_hot_labels = tf.one_hot(labels, num_items, dtype=tf.float32)
            print(labels.shape, logits.shape)
#             print(labels, logits)
            cross_entropy = tf.nn.sigmoid_cross_entropy_with_logits(labels=labels, logits=logits) # tf.expand_dims()
            mean_loss = tf.reduce_mean(cross_entropy)
#             print(cross_entropy, mean_loss)
            if mode == tf.estimator.ModeKeys.TRAIN:
                optimizer = tf.compat.v1.train.AdamOptimizer()#GradientDescentOptimizer(learning_rate)
                # Input to reshape is a tensor with 32 values, but the requested shape has 1
    #             train_op = optimizer.minimize(cross_entropy, tf.compat.v1.train.get_or_create_global_step())
                
                train_op = optimizer.minimize(mean_loss, tf.compat.v1.train.get_or_create_global_step())
                estimatorSpec = tf.estimator.EstimatorSpec(mode, loss=mean_loss, train_op=train_op)
                return estimatorSpec

            else: #mode == tf.estimator.ModeKeys.EVAL
                def metric_fn(labels, predictions):
                    return {"accuracy":tf.compat.v1.metrics.accuracy(labels, predictions)
                           }
                eval_metrics = metric_fn(labels, predictions)
                estimatorSpec = tf.estimator.EstimatorSpec(mode=mode, loss=mean_loss, eval_metric_ops = eval_metrics)
                return estimatorSpec

    return model_fn

def input_fn_builder(input_features, labels=None, batch_size=64, epoch_num=1, is_training=False, shuffle_buffer_size=None, seed=None):
    # input_features = {'cont_features':[num_examples, num_cont_features], 
    #                   'embedding_features':{'space_id':{'fixed':[num_examples],
    #                                                 'ragged':{'indices':[num_examples], 'dense_shape':[row, col]},
#                                                     'truncated':{'fixed_ids':[num_examples], 'true_lengths':[num_examples]}
    #                                                }}}
    def input_fn():
        features = {}
#         print(input_features)
        num_examples = input_features['num_examples']
        if 'cont_features' in input_features:
            features['cont_features'] = tf.constant(input_features['cont_features'], dtype=tf.float32)
            
        if 'embedding_features' in input_features:
            features['embedding_features'] = {}
            for space_id, embedding in input_features['embedding_features'].items():
                fixed_ragged = {}
                if 'univalent' in embedding:
                    fixed_ragged['univalent'] = tf.constant(embedding['univalent'], dtype=tf.int32)
                    
                if 'multivalent' in embedding:
                    ragged_indices = embedding['multivalent']['indices']
                    indices = [[i, j] for i in range(num_examples) for j in range(len(ragged_indices[i]))]
                    values = [x for row in ragged_indices for x in row]
                    dense_shape = embedding['multivalent']['dense_shape'] # [num_examples, embedding['length']]
                    sp_indexes = tf.SparseTensor(indices=indices, values=values, dense_shape=dense_shape)
                    fixed_ragged['multivalent'] = sp_indexes
                
                features['embedding_features'][space_id] = fixed_ragged
                
        if labels is None:
            data_set = tf.data.Dataset.from_tensor_slices(features)
        else:
            data_set = tf.data.Dataset.from_tensor_slices((features, labels))
        if is_training:
            print(f'epoch {epoch_num}')
            data_set = data_set.repeat(epoch_num)
            if shuffle_buffer_size is None:
                data_set = data_set.shuffle(buffer_size=num_examples, seed=seed)
            else:
                data_set = data_set.shuffle(buffer_size=shuffle_buffer_size, seed=seed)
        
        data_set = data_set.prefetch(buffer_size=128)
        data_set = data_set.batch(batch_size=batch_size)

        return data_set
    return input_fn


# In[22]:


def save_meta_data(num_example, num_cont_features, embedding_shapes, meta_file_path = 'train.meta'):
    meta_map = {'embedding_space': embedding_shapes}
    meta_map['num_example'] = num_example
    meta_map['num_cont_features'] = num_cont_features
    
    with open(meta_file_path, "w") as file:
        json.dump(meta_map, file)
        
def load_meta_data(meta_file_path = 'train.meta'):
    with open(meta_file_path, "r") as file:
        return json.load(file)
    
def truncate_and_pad_seq(ids, max_length, paddings=0):
    true_length = len(ids)
    fixed_ids = ids[:max_length]
    while len(fixed_ids) < max_length:
        fixed_ids.append(paddings)
        
    return fixed_ids, true_length

def truncate_and_pad_seqs(ids, max_length, paddings=0):
    fixed_ids = []
    true_lengths = []
    for row in ids:
        fixed_id, true_length = truncate_and_pad_seq(row, max_length, paddings)
        fixed_ids.append(fixed_id)
        true_lengths.append(true_length)
    return fixed_ids, true_lengths
    
def df_to_input_features(df, embedding_space, cols_del, meta_file_path = 'train.meta', truncated_cols={'utvId_past_emb':50}):
    input_features = {}
#   embedding_space = [space_id: {'embedding_size':embedding_size, 'encoder':encoder,
#                                   'univalent':col, 'multivalent':col}]
#   truncated_cols = {'col':max_seq_size}

    cont_cols = np.setdiff1d(df.columns, cols_del, assume_unique=True)
    other_cols = []
#     embedding_cols = {'space_id':{'fixed': col, 'ragged': col, 'length': n_class }}
    embedding_features = {}
    embedding_shapes = {}
    for space_id, space in embedding_space.items():
        embedding_features[space_id] = {}
        num_classes = len(space['encoder'].classes_)
        embedding_shapes[space_id] = {'embedding_shape':[num_classes, space['embedding_size']]}
        if 'univalent' in space:
            col = space['univalent']
            embedding_features[space_id]['univalent'] = df[col].values
            other_cols.append(col)
            embedding_shapes[space_id]['univalent'] = col
        if 'multivalent' in space:
            col = space['multivalent']
            indices = df[col].values
            other_cols.append(col)
            if truncated_cols is not None and col in truncated_cols:
                max_seq_size = truncated_cols[col]
                fixed_ids, true_lengths = truncate_and_pad_seqs(indices, max_seq_size)
                embedding_features[space_id]['truncated'] = {'fixed_ids':fixed_ids, 'true_lengths':true_lengths}
                embedding_shapes[space_id]['truncated'] = max_seq_size
            else:
                embedding_features[space_id]['multivalent'] = {'indices':indices, 'dense_shape':[df.shape[0], num_classes]}
                embedding_shapes[space_id]['multivalent'] = col
        
    if len(embedding_features) > 0:
        input_features['embedding_features'] = embedding_features
    cont_cols = np.setdiff1d(cont_cols, other_cols, assume_unique=True)
    print(cont_cols)
    if len(cont_cols) > 0:
        input_features['cont_features'] = df[cont_cols].values
    input_features['num_examples'] = df.shape[0]
    save_meta_data(input_features['num_examples'], df[cont_cols].values.shape[1], embedding_shapes, meta_file_path=meta_file_path)
    return input_features, embedding_shapes


# In[14]:


# df_train1 = pd.concat([df_train, df_test], axis=0)
cols_del = ['uid','event_time_now','utvId_now','collect_day_now','utvId','y', 'uploadtime']
input_features_train, embedding_shapes = df_to_input_features(df_train, embedding_space=embedding_space, cols_del=cols_del)
y_train = df_train[['y']].values.astype(np.float32)


# In[30]:


input_features_train['embedding_features']['utvid']['truncated']['true_lengths'][3]


# In[31]:


# input_features_train['embedding_features']['utvid']['truncated']['fixed_ids'][3]


# In[18]:


# features = {'cont_features': VarLenFeature, space_id+'_fixed': }
def input_features_to_tfrecords(input_features, labels = None, filename = 'train.tfrecords'):
    # input_features = {'cont_features':[num_examples, num_cont_features], 
    #                   'embedding_features':{'space_id':{'univalent':[num_examples], 
    #                                                 'multivalent':{'indices':[num_examples], 'dense_shape':[row, col]} 
    #                                                }},
    #                   'num_examples': num_examples}
    num_examples = input_features['num_examples']
    cont_features = input_features['cont_features']
    embedding_features = input_features['embedding_features']
    
    writer = tf.io.TFRecordWriter(filename)
    for i in range(num_examples):
        if i % 50000 == 0:
            utils.print_with_datetime(f'Finished {i}')
        
        cont_feat = tf.train.Feature(float_list=tf.train.FloatList(value=cont_features[i]))
        feat_map = {'cont_features': cont_feat}
        for space_id, space in embedding_features.items():
            feat_list = []
            if 'univalent' in space:
                feat_map[space_id+'_univalent'] = tf.train.Feature(int64_list=tf.train.Int64List(value=[space['univalent'][i]]))
            if 'multivalent' in space:
                feat_map[space_id+'_multivalent'] = tf.train.Feature(int64_list=tf.train.Int64List(value=space['multivalent']['indices'][i]))
            if 'truncated' in space:
                feat_map[space_id+'_fixed_id'] = tf.train.Feature(int64_list=tf.train.Int64List(value=space['truncated']['fixed_ids'][i]))
                feat_map[space_id+'_true_length'] = tf.train.Feature(int64_list=tf.train.Int64List(value=[space['truncated']['true_lengths'][i]]))
        if labels is not None:
            feat_map['labels'] = tf.train.Feature(float_list = tf.train.FloatList(value=labels[i]))
        
        features = tf.train.Features(feature=feat_map)
        tf_example = tf.train.Example(features=features)
        writer.write(tf_example.SerializeToString())
    writer.close()

input_features_to_tfrecords(input_features_train, y_train)


# In[26]:


def get_decode_map(embedding_space, is_training=False, len_cont_feats=11):
    decode_map = {'cont_features': tf.io.FixedLenFeature([len_cont_feats], tf.float32)}
    for space_id, space in embedding_space.items():
        if 'univalent' in space:
            decode_map[space_id+'_univalent'] = tf.io.FixedLenFeature([], tf.int64)
        if 'multivalent' in space:
            decode_map[space_id+'_multivalent'] = tf.io.VarLenFeature(tf.int64)
        if 'truncated' in space:
            max_seq_size = space['truncated']
            decode_map[space_id+'_fixed_id'] = tf.io.FixedLenFeature([max_seq_size], tf.int64)
            decode_map[space_id+'_true_length'] = tf.io.FixedLenFeature([], tf.int64)
            
    if is_training:
        decode_map['labels'] = tf.io.FixedLenFeature([1], tf.float32)
    return decode_map


def example_to_model_input(example, embedding_space):
#   features = {'cont_features':tensor, 
#               'embedding_features':{space_id: {'univalent':tensor, 'multivalent':sparse_tensor}}}
#     embedding_space=meta['embedding_space']
#     attention=meta['attention']
    # {attention:{query:{key:value}}}, e.g. attention:{'utvid_univalent':{'utvid_multivalent':'utvid_multivalent'}}}
    # attention={'query':'utvid_univalent', 'key':'utvid_multivalent'}
    features = {}
    if 'cont_features' in example:
        features['cont_features'] = example['cont_features']
    if embedding_space:
        features['embedding_features'] = {}
    for space_id, values in embedding_space.items():
        features['embedding_features'][space_id] = {}
#         print(space_id, values)
        if 'univalent' in values:
            features['embedding_features'][space_id]['univalent'] = example[space_id+'_univalent']
        if 'multivalent' in values:
            features['embedding_features'][space_id]['multivalent'] = example[space_id+'_multivalent']
        if 'truncated' in values:
            features['embedding_features'][space_id]['fixed_id'] = example[space_id+'_fixed_id']
            features['embedding_features'][space_id]['true_length'] = example[space_id+'_true_length']
    labels = None
    if 'labels' in example:
        labels = example['labels']

    return features, labels

def parse_nested_dict(example_proto, meta, is_training=False):
    decode_map = get_decode_map(meta['embedding_space'], is_training, meta['num_cont_features'])
    example = tf.io.parse_single_example(example_proto, features=decode_map)

    model_input_features, labels = example_to_model_input(example, meta['embedding_space'])
    if labels is not None:
        return model_input_features, labels
    else:
        return model_input_features

def input_fn_builder_tf(meta, tfrecord_path='train.tfrecords',  batch_size=64, epoch_num=1, is_training=False, 
                        shuffle_buffer_size=None, seed=None):
    # input_features = {'cont_features':[num_examples, num_cont_features], 
    #                   'embedding_features':{'space_id':{'fixed':[num_examples],
    #                                                 'ragged':{'indices':[num_examples], 'dense_shape':[row, col]}
    #                                                }}}
    def input_fn():
        data_set = tf.data.TFRecordDataset(tfrecord_path)
        data_set = data_set.map(lambda x: parse_nested_dict(x, meta, is_training))
        if is_training:
            print(f'epoch {epoch_num}')
            data_set = data_set.repeat(epoch_num)
            if shuffle_buffer_size is None:
                data_set = data_set.shuffle(buffer_size=train_meta['num_example'], seed=seed)
            else:
                data_set = data_set.shuffle(buffer_size=shuffle_buffer_size, seed=seed)
        
        data_set = data_set.batch(batch_size=batch_size).prefetch(buffer_size=1)

        return data_set
    return input_fn


# In[21]:


train_meta


# In[28]:


def att_score(query, key):
    # score(query, key) = W_2*PReLU(W_1*concate([query, key, query-key, query*key]))
    score = tf.concat([query, key, key-query, key*query], axis=-1) 
    score = tf.keras.layers.Dense(36, activation=tf.nn.relu)(score)
    score = tf.keras.layers.Dense(1)(score)
    return score

def attention_module(query, keys, masks, values=None):
    '''
    queries:    [B, H]
    keys:       [B, T, H]
    masks:      [B, T]
    '''
    if values is None:
        values = keys
        
    emb_size = query.shape[-1]
#     weights = att_score(query, keys) # [B, T, 1]
    weights = key*query # [B, T, H]
    weights = tf.reduce_sum(weights, -1) # [B, T, 1]
    neg_inf = -2 ** 32 + 1
    weights = tf.where(masks, weights, neg_inf)
    # scaled dot product
    weights = weights / (emb_size ** 0.5) # 我觉得可以在att_score的query*key中scale, 其它不用scale
    
    return values * weights


# In[31]:


epoch_num = 1 #25
batch_size = 64
seed = 2019
num_layers =  [512, 256, 128] #[512,256,128] # [256,128] # 0.7153 0.747 
train_meta = load_meta_data()
embedding_shapes = train_meta['embedding_space']
model_fn = model_fn_builder(embedding_shapes=embedding_shapes, num_layers=num_layers)

# train_input_fn = input_fn_builder(input_features_train, y_train, batch_size, epoch_num, is_training=True, seed=seed)
train_input_fn = input_fn_builder_tf(train_meta, batch_size=batch_size, epoch_num=epoch_num, is_training=True, seed=seed)
eval_input_fn = train_input_fn
# train_input_fn = input_fn_builder(X_train, cont_features_train, uid_features_train,y_train, 
#                                   len(encoder.classes_),batch_size, epoch_num, is_training=True, seed=seed)
# eval_input_fn = input_fn_builder(X_eval, cont_features_eval, uid_features_test, y_eval, len(encoder.classes_), batch_size) 
# eval_input_fn = input_fn_builder(X_eval_clean, cont_features_eval_clean, uid_features_test_clean, y_eval_clean, num_items, batch_size) 

config = tf.estimator.RunConfig(
    model_dir="rank_dcn/",
    tf_random_seed=seed,
    save_checkpoints_steps=30000,
    keep_checkpoint_max=5,
    log_step_count_steps=5000
)

estimator = tf.estimator.Estimator(model_fn=model_fn, config=config)

# train_spec = tf.estimator.TrainSpec(
#     input_fn=train_input_fn,
#     max_steps=1000
# )

train_result = estimator.train(input_fn=train_input_fn) #, steps=100 , max_steps=10200
# eval_result = estimator.evaluate(input_fn=eval_input_fn)
# eval_result = estimator.train_and_evaluate()
utils.print_with_datetime("***** Eval results *****")
# for key in sorted(eval_result.keys()):
#     print(key + '='+ str(eval_result[key]))


# In[33]:


def get_recall_topk_user_time_item(df, encoder, k = 30):
    recall_pred_probs = pd.read_csv('df_pred_probs2.csv',header=None).values
    top_k_utvid = tf.nn.top_k(recall_pred_probs, k=k).indices.numpy()
    print('top_k_utvid',top_k_utvid.shape) # top_k_utvid (43644, 30)
    top_k_utvid = top_k_utvid.flatten()
    arr_topk = np.empty_like(df, shape=(top_k_utvid.shape[0], df.shape[1]))
    print('arr_topk', arr_topk.shape) # arr_topk (1309320, 10)
    print(len(df), k) # 1198264 30
    for i in range(len(df)):
        if i % 5000 == 0:
            utils.print_with_datetime(f'Finished getting {i} top {k} data')
        offset = i * k
        for j in range(k):
            o = offset + j
            for c in range(len(df.columns)):
                arr_topk[o,c] = df.iloc[i,c]
    df_topk = pd.DataFrame(arr_topk, columns=df.columns)
    df_topk['utvId_now'] = encoder.back_transform(top_k_utvid)
    return df_topk

def get_topk_data(df):
    user_time_item = {}
    user_time_item['categorical'] = {'univalent':{'embedding': [{'space_id':'utvid', 'col':'utvId_now'}]}}
    user_time_item['reference'] = {'user_time':{'uid':'uid', 'event_time_now':'event_time_now'},
                                   'item':{'utvId_now':'utvId'}}
    nested_data = [{'data':df, 'features': {'user_time_item':user_time_item}}]

    return nested_data

utvid_encoder = embedding_space['utvid']['encoder']
k=30
df_train_clean, df_test_clean = split_fn(df_user_time_item) # [df_user_time_item['y']==1]
df_test_clean = df_test_clean[df_test_clean['utvId_now'].isin(df_train_clean['utvId_now'])]
df_topk = get_recall_topk_user_time_item(df_test_clean, utvid_encoder, k=k)
del df_topk['y']
topk_data = get_topk_data(df_topk)
df_pred, _, _, _, _, _ = process_data(topk_data, need_scale=True, cont_cols=cont_cols, scaler=scaler,embedding_space=embedding_space, pk=pk)
col_fn_map = {'now_uploadtime': lambda x: x['event_time_now']//1e3 - datetime.datetime.strptime(str(x['uploadtime']), '%Y-%m-%d %H:%M:%S').timestamp(),
              'example_age': lambda x: 0, 
              'example_age_square': lambda x: 0,
              'example_age_rootsquare': lambda x: 0}
df_pred, _ = postprocess(df_pred, col_fn_map, scaler=post_scaler)


# In[34]:


test_tfrecord_path = 'test.tfrecords'
test_meta_path = 'test.meta'
input_features_pred, _ = df_to_input_features(df_pred, embedding_space=embedding_space, cols_del=cols_del, meta_file_path=test_meta_path)
input_features_to_tfrecords(input_features_pred, filename=test_tfrecord_path)


# In[35]:


pred_meta = load_meta_data(test_meta_path)
# pred_input_fn = input_fn_builder_tf(pred_meta, tfrecord_path=test_tfrecord_path, batch_size=512)
pred_input_fn = input_fn_builder(input_features_pred, batch_size=512)

pred = estimator.predict(input_fn=pred_input_fn)
pred_probs = []
for output in pred:
#     print(output)
    pred_probs.append(output['probabilities']) # np.contecate
# pred_probs = pred_probs.reshape((X_topk.shape[0],1))
# pred_probs = pred_probs.reshape((pred_probs.shape[0]//k, k))
# pred_probs = pred_probs.reshape(pred_probs.shape[0] * k)
pred_probs = np.array(pred_probs)


# In[40]:


utvid_encoder = embedding_space['utvid']['encoder']
target_ids_eval_clean = np.array(utvid_encoder.transform(df_test_clean['utvId_now']))
# target_ids_topk = df_pred['utvId_now_emb'].values
target_ids_topk =  np.array(utvid_encoder.transform(df_pred['utvId_now'].values))
n = pred_probs.shape[0]//k
sparse_indices = [[i, target_ids_topk[i*k + j]] for i in range(n) for j in range(k) ]
num_items = embedding_shapes['utvid']['embedding_shape'][0] 
pred_probs_n_hot = sparseTensor(sparse_indices, pred_probs, (n, num_items), dtype=np.float16)
pred_probs_n_hot.shape


# In[41]:


for k in [1,5,10,20,30]: #,50,100
    top_k_acc = tf.metrics.top_k_categorical_accuracy(tf.one_hot(target_ids_eval_clean, num_items), pred_probs_n_hot, k=k).numpy()
    utils.print_with_datetime(f'top {k} accuracy is {top_k_acc.sum() / top_k_acc.shape[0]}')


# In[108]:


for topk in [1,5,10,20,30]:
    max_k_preds = pred_probs_n_hot.argsort(axis=1)[:, -topk:][:, ::-1] 
    match_array = np.logical_or.reduce(max_k_preds==target_ids_eval_clean.reshape((target_ids_eval_clean.shape[0],1)), axis=1) 
    utils.print_with_datetime(f'top {topk} accuracy is {match_array.sum() / match_array.shape[0]}')


# In[83]:


for k in [1,5,10,20,30]: #,50,100
    max_k_preds = pred_probs_n_hot.argsort(axis=1)[:, -k:][:, ::-1] 
    match_array = np.logical_or.reduce(max_k_preds==target_ids_eval_clean.reshape((target_ids_eval_clean.shape[0],1)), axis=1) 
    utils.print_with_datetime(f'top {k} accuracy is {match_array.sum() / match_array.shape[0]}')


# In[23]:


recall_pred_probs = pd.read_csv('df_pred_probs2.csv',header=None).values
for k in [1,5,10,20,30,50,100]:
    max_k_preds = recall_pred_probs.argsort(axis=1)[:, -k:][:, ::-1] 
    match_array = np.logical_or.reduce(max_k_preds==target_ids_eval_clean.reshape((target_ids_eval_clean.shape[0],1)), axis=1) 
    print(f'top {k} accuracy is {match_array.sum() / match_array.shape[0]}')

