import pandas as pd
import os
from Modules.utils import pad_history, to_pickled_df

def merge_consec_bookings(group):
    merged_rows = []
    current_row = None
    for _, row in group.iterrows():
        if current_row is None:
            current_row = row
        elif row['city_id'] == current_row['city_id'] and row['checkin'] == current_row['checkout']:
            current_row['checkout'] = row['checkout']
        else:
            merged_rows.append(current_row)
            current_row = row
    merged_rows.append(current_row)
    return pd.DataFrame(merged_rows)

def set_rewards_toppop(filename, save_name, pop_ratio=0.1):
    df = pd.read_csv(filename)
    ch_ind = int(df['city_id'].nunique() * pop_ratio)
    toppop_cities = df.city_id.value_counts().index[:ch_ind]
    df['is_buy'] = (df['city_id'].isin(toppop_cities)).astype(int)
    df.to_csv(save_name, index=False)
    return df

def set_rewards_most_time(filename, save_name):
    df = pd.read_csv(filename)
    df['checkin'] = pd.to_datetime(df['checkin'])
    df['checkout'] = pd.to_datetime(df['checkout'])
    df['time_at'] = (df['checkout'] - df['checkin']).dt.days
    max_time_at_indices = df.groupby('utrip_id')['time_at'].idxmax()
    df['is_buy'] = 0
    df.loc[max_time_at_indices, 'is_buy'] = 1
    df.to_csv(save_name, index=False)
    return df

def get_statistics(sampled_sessions : pd.DataFrame, train_sessions : pd.DataFrame, length=10):
    # data_directory = 'data'

    length = length

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

    dic={'state_size':[length],'item_num':[pad_item]}
    data_statis=pd.DataFrame(data=dic)

    return reply_buffer, data_statis

def preprocess_booking(filename, save_name, itm2idx=None):
    df = pd.read_csv(filename)
    df.columns = ['session_id', 'timestamp', 'checkout', 'item_id', 'time_at', 'is_buy']
    df = df.drop(['checkout', 'time_at'], axis=1)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    # df['valid_session'] = df.session_id.map(df.groupby('session_id')['item_id'].size() > 2)
    # df = df.loc[df.valid_session].drop('valid_session', axis=1)
    if itm2idx is None:
        itm2idx = {k: i for i, k in enumerate(set(df['item_id'].unique()))}
    idx2itm = {v: k for k, v in itm2idx.items()}
    df['item_id'] = df['item_id'].map(itm2idx)
    sorted_df = df.sort_values(by=['session_id', 'timestamp'])
    sorted_df.to_csv(save_name, index=None)
    return sorted_df, itm2idx, idx2itm

def create_pop_dict(data_dir, filename='sorted_events.df'):
    data_directory = data_dir
    if filename[-1] == 'f':
        replay_buffer_behavior = pd.read_pickle(os.path.join(data_directory, filename))
    else:
        replay_buffer_behavior = pd.read_csv(data_directory + filename)
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
