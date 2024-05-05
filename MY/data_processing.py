import typing
import pandas as pd
import os
from MY.utils import to_pickled_df, pad_history
import numpy as np
from sklearn.preprocessing import LabelEncoder

def shorten_sessions(sorted_session, n_sessions):
    all_sessions = sorted_session['session_id'].unique()
    chosen_sess = np.random.choice(all_sessions, size=n_sessions, replace=False)
    chosen = sorted_session.loc[sorted_session['session_id'].isin(chosen_sess)].copy()
    
    items_dict = {k: i for i, k in enumerate(set(chosen['item_id'].unique()))}
    for idx, row in chosen.iterrows():
        row['item_id'] = items_dict[row['item_id']]
    print('unique:', chosen['item_id'].nunique(), 'min:', chosen['item_id'].min(), 'max:', chosen['item_id'].max())
    return chosen

def preprocess_books(filename, is_buy_progress_criterion=65):
    df = pd.read_csv(filename, header=0)
    df.columns = ['session_id', 'item_id', 'progress', 'rating','timestamp']
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df['timestamp'] = (df['timestamp'] - df['timestamp'].min()).dt.days
    df['valid_session'] = df.session_id.map(df.groupby('session_id')['item_id'].size() > 2)
    df = df.loc[df.valid_session].drop('valid_session', axis=1)
    ##########remove items with <=2 interactions
    df['valid_item'] = df.item_id.map(df.groupby('item_id')['session_id'].size() > 2)
    df = df.loc[df.valid_item].drop('valid_item', axis=1)
    
    df['is_buy'] = (df['progress'] > is_buy_progress_criterion).astype(int)
    rated_df = df[df['rating'].notnull()]
    df = df.drop(['progress', 'rating'], axis=1)
    sorted_df = df.sort_values(by=['session_id', 'timestamp'])
    sorted_df.to_csv('books_dataset/sorted_events.csv', index=None, header=True)
    print(df)
    return df, rated_df


def sample_data(data_dir : str, name : str, to_pickle=False):
    
    event_df = pd.read_csv(os.path.join(data_dir, 'events.csv'), header=0)
    event_df.columns = ['timestamp','session_id','behavior','item_id','transid']
    ###remove transid column
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

def create_pop_dict(data_dir='data'):
    data_directory = data_dir
    replay_buffer_behavior = pd.read_pickle(os.path.join(data_directory, 'sorted_events.df'))
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
    