import typing
import pandas as pd
import os
from Modules.utils import to_pickled_df, pad_history
import numpy as np
from sklearn.preprocessing import LabelEncoder

def shorten_sessions(sorted_session, n_sessions):
    all_sessions = sorted_session['session_id'].unique()
    chosen_sess = np.random.choice(all_sessions, size=n_sessions, replace=False)
    chosen = sorted_session.loc[sorted_session['session_id'].isin(chosen_sess)].copy()
    
    itm2idx = {k: i for i, k in enumerate(set(chosen['item_id'].unique()))}
    chosen['item_id'] = chosen['item_id'].map(itm2idx)
    print('unique:', chosen['item_id'].nunique(), 'min:', chosen['item_id'].min(), 'max:', chosen['item_id'].max())
    return chosen, itm2idx

def preprocess_books(filename):
    df = pd.read_csv(filename, header=0)
    df.columns = ['session_id', 'item_id', 'progress', 'rating','timestamp']
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df['timestamp'] = (df['timestamp'] - df['timestamp'].min()).dt.days
    df['valid_session'] = df.session_id.map(df.groupby('session_id')['item_id'].size() > 2)
    df = df.loc[df.valid_session].drop('valid_session', axis=1)
    ##########remove items with <=2 interactions
    df['valid_item'] = df.item_id.map(df.groupby('item_id')['session_id'].size() > 2)
    df = df.loc[df.valid_item].drop('valid_item', axis=1)
    sorted_df = df.sort_values(by=['session_id', 'timestamp'])
    sorted_df.to_csv('books_data/init_sorted_events.csv', index=None, header=True)
    return df

def set_books_reward_progress(df_books, is_buy_progress_criterion=75):
    df = df_books.copy()
    df['is_buy'] = (df['progress'] >= is_buy_progress_criterion).astype(int)
    df = df.drop(['progress', 'rating'], axis=1)
    df = df.sort_values(by=['session_id', 'timestamp'])
    df.to_csv('books_data/progress_reward_books.csv', index=None, header=True)
    return df

def set_books_reward_rating(df_books, is_buy_rating_criterion=4.0):
    df = df_books.copy()
    df['is_buy'] = (df['rating'] >= is_buy_rating_criterion).astype(int)
    df = df.drop(['progress', 'rating'], axis=1)
    df = df.sort_values(by=['session_id', 'timestamp'])
    df.to_csv('books_data/rating_reward_books.csv', index=None, header=True)
    return df

def func_auth(group):
    new = []
    author_set = set()
    for idx, row in group.iterrows():
        cur_a_set = set(row['authors'])
        if bool(author_set & cur_a_set):
            row['is_buy'] = 1
        author_set.update(cur_a_set)
        new.append(row)
    return pd.DataFrame(new)


def set_books_reward_author(df_books, item_info_df):
    df = df_books.copy()
    merged_df = df.merge(item_info_df[['id', 'authors']], how='left', left_on='item_id', right_on='id').drop(['id'], axis=1)
    merged_df['authors'] = merged_df['authors'].str.split(',')
    merged_df['is_buy'] = 0
    merged_df = merged_df.groupby('session_id').apply(func_auth).reset_index(drop=True)
    merged_df = merged_df.drop(['authors', 'progress', 'rating'], axis=1)
    merged_df = merged_df.sort_values(by=['session_id', 'timestamp'])
    merged_df.to_csv('books_data/author_reward_books.csv', index=None, header=True)
    return merged_df

def func_genre(group):
    new = []
    genre_set = set()
    for idx, row in group.iterrows():
        cur_g_set = set(row['genres'])
        if bool(genre_set & cur_g_set):
            row['is_buy'] = 1
        genre_set.update(cur_g_set)
        new.append(row)
    return pd.DataFrame(new)


def set_books_reward_genre(df_books, item_info_df):
    df = df_books.copy()
    merged_df = df.merge(item_info_df[['id', 'genres']], how='left', left_on='item_id', right_on='id').drop(['id'], axis=1)
    merged_df['genres'] = merged_df['genres'].str.split(',')
    merged_df['is_buy'] = 0
    merged_df = merged_df.groupby('session_id').apply(func_genre).reset_index(drop=True)
    merged_df = merged_df.drop(['genres', 'progress', 'rating'], axis=1)
    merged_df = merged_df.sort_values(by=['session_id', 'timestamp'])
    merged_df.to_csv('books_data/genres_reward_books.csv', index=None, header=True)
    return merged_df



def sample_data(data_dir : str, name : str, to_pickle=False):
    event_df = pd.read_csv(os.path.join(data_dir, 'events.csv'), header=0)
    event_df.columns = ['timestamp','session_id','behavior','item_id','transid']
    event_df =event_df[event_df['transid'].isnull()]
    event_df = event_df.drop('transid',axis=1)
    ##########remove users with <=2 interactions
    event_df['valid_session'] = event_df.session_id.map(event_df.groupby('session_id')['item_id'].size() > 2)
    event_df = event_df.loc[event_df.valid_session].drop('valid_session', axis=1)
    ##########remove items with <=2 interactions
    event_df['valid_item'] = event_df.item_id.map(event_df.groupby('item_id')['session_id'].size() > 2)
    event_df = event_df.loc[event_df.valid_item].drop('valid_item', axis=1)
    ######## transform to ids
    item_encoder = LabelEncoder()
    session_encoder= LabelEncoder()
    behavior_encoder=LabelEncoder()
    event_df['item_id'] = item_encoder.fit_transform(event_df.item_id)
    event_df['session_id'] = session_encoder.fit_transform(event_df.session_id)
    event_df['behavior']=behavior_encoder.fit_transform(event_df.behavior)
    ###########sorted by user and timestamp
    event_df['is_buy']=1-event_df['behavior']
    event_df = event_df.drop('behavior', axis=1)
    sorted_events = event_df.sort_values(by=['session_id', 'timestamp'])

    sorted_events.to_csv('data/sorted_events.csv', index=None, header=True)
    
    if to_pickle:
        to_pickled_df(data_dir, sorted_events=sorted_events)
    
    return sorted_events


def split_data(sorted_events : pd.DataFrame, pickle=False, data_dir='data', sessions_name='sorted_events.df'):
    # sampled_buys=pd.read_pickle(os.path.join(data_directory, 'sampled_buys.df'))
    #
    # buy_sessions=sampled_buys.session_id.unique()
    if pickle:
        sorted_events = pd.read_pickle(os.path.join(data_dir, sessions_name))
    
    total_sessions=sorted_events.session_id.unique()
    np.random.shuffle(total_sessions)

    fractions = np.array([0.8, 0.1, 0.1])
    # split into 3 parts
    train_ids, val_ids, test_ids = np.array_split(
        total_sessions, (fractions[:-1].cumsum() * len(total_sessions)).astype(int))

    train_sessions=sorted_events[sorted_events['session_id'].isin(train_ids)]
    val_sessions=sorted_events[sorted_events['session_id'].isin(val_ids)]
    test_sessions=sorted_events[sorted_events['session_id'].isin(test_ids)]

    if pickle:
        to_pickled_df(data_dir, sampled_train=train_sessions)
        to_pickled_df(data_dir, sampled_val=val_sessions)
        to_pickled_df(data_dir,sampled_test=test_sessions)
        
    return train_sessions, val_sessions, test_sessions



def get_statistics(sampled_sessions : pd.DataFrame, train_sessions : pd.DataFrame, pickle=False, data_dir='data', sessions_name='sorted_events.df', train_name='sampled_train.df'):
    # data_directory = 'data'

    length = 10

    if pickle:
    #  reply_buffer = pd.DataFrame(columns=['state','action','reward','next_state','is_done'])
        sampled_sessions=pd.read_pickle(os.path.join(data_dir, sessions_name))
        train_sessions = pd.read_pickle(os.path.join(data_dir, train_name))
    
    item_ids=sampled_sessions.item_id.unique()
    pad_item=len(item_ids)

    groups=train_sessions.groupby('session_id')
    ids=train_sessions.session_id.unique()

    state, len_state, action, is_buy, next_state, len_next_state, is_done = [], [], [], [], [],[],[]

    for id in ids:
        group=groups.get_group(id)
        history=[]
        for index, row in group.iterrows():
            s=list(history)
            len_state.append(length if len(s)>=length else 1 if len(s)==0 else len(s))
            s=pad_history(s,length,pad_item)
            a=row['item_id']
            is_b=row['is_buy']
            state.append(s)
            action.append(a)
            is_buy.append(is_b)
            history.append(row['item_id'])
            next_s=list(history)
            len_next_state.append(length if len(next_s)>=length else 1 if len(next_s)==0 else len(next_s))
            next_s=pad_history(next_s,length,pad_item)
            next_state.append(next_s)
            is_done.append(False)
        is_done[-1]=True

    dic={'state':state,'len_state':len_state,'action':action,'is_buy':is_buy,'next_state':next_state,'len_next_states':len_next_state,
         'is_done':is_done}
    
    reply_buffer=pd.DataFrame(data=dic)
    if pickle:
        to_pickled_df(data_dir, replay_buffer=reply_buffer)

    dic={'state_size':[length],'item_num':[pad_item]}
    data_statis=pd.DataFrame(data=dic)
    if pickle:
        to_pickled_df(data_dir,data_statis=data_statis)
    return reply_buffer, data_statis

def create_pop_dict(data_dir='data', filename='sorted_events.df'):
    data_directory = data_dir
    if filename[-1] == 'f':
        replay_buffer_behavior = pd.read_pickle(os.path.join(data_directory, filename))
    else:
        replay_buffer_behavior = pd.read_csv(os.path.join(data_directory, filename))
    total_actions=replay_buffer_behavior.shape[0]
    pop_dict={}
    for index, row in replay_buffer_behavior.iterrows():
        action=row['item_id']
        if action in pop_dict:
            pop_dict[action]+=1
        else:
            pop_dict[action]=1
        if index%100000==0:
            print (index/100000)
    for key in pop_dict:
        pop_dict[key]=float(pop_dict[key])/float(total_actions)

    with open(os.path.join(data_directory, 'pop_dict.txt'), 'w') as f:
        f.write(str(pop_dict))

def df_create_pop_dict(df_books):
    replay_buffer_behavior = df_books
    total_actions=replay_buffer_behavior.shape[0]
    pop_dict={}
    for index, row in replay_buffer_behavior.iterrows():
        action=row['item_id']
        if action in pop_dict:
            pop_dict[action]+=1
        else:
            pop_dict[action]=1
        if index%100000==0:
            print (index/100000)
    for key in pop_dict:
        pop_dict[key]=float(pop_dict[key])/float(total_actions)

    with open('pop_dict.txt', 'w') as f:
        f.write(str(pop_dict))
