import tensorflow as tf
import numpy as np
import pandas as pd
import os
import argparse
from collections import deque
from MY.utils import pad_history, calculate_hit, extract_axis_1, calculate_off
#from NextItNetModules import *
#from MY.SASRecModules import *

import trfl
from trfl import indexing_ops


import tensorflow as tf
import tensorflow_probability as tfp

import tensorflow as tf
import tensorflow_probability as tfp

import tensorflow as tf

import tensorflow as tf
import tensorflow_probability as tfp

import copy

# class QNetwork(tf.Module):
#     def __init__(self, item_num, state_size, arg_dict, configuration='DQN', sequential_model='GRU', pretrain=False, name='DQNetwork'):
#         super(QNetwork, self).__init__(name=name)
        
#         self.batch_size = 256 if 'batch_size' not in arg_dict else arg_dict['batch_size']
#         self.reward_negative = 1 if 'r_negative' not in arg_dict else arg_dict['r_negative']
#         self.state_size = state_size
#         self.item_num = int(item_num)
#         self.learning_rate = 0.001 if 'lr' not in arg_dict else arg_dict['lr']
#         self.hidden_size = 64 if 'hidden_factor' not in arg_dict else arg_dict['hidden_factor']
#         self.neg = 10 if 'neg' not in arg_dict else arg_dict['neg']
#         self.pretrain = pretrain
#         self.model = sequential_model
#         self.configuration = configuration 
        
#         self.is_training = tf.Variable(False, trainable=False, dtype=tf.bool, name='is_training')
        
#         # for SASREC
#         self.dropout_rate = 0.1
#         self.num_blocks = 1
#         self.num_heads = 1
        
#         # for SA2C
#         if self.configuration == 'SA2C':
#             self.smooth = 0.0 if 'smooth' not in arg_dict else arg_dict['smooth']
#             self.learning_rate2 = 0.001 if 'lr2' not in arg_dict else arg_dict['lr2']
#             self.clip = 0.0 if 'clip' not in arg_dict else arg_dict['clip']
        
#         with self.name_scope:
#             self.all_embeddings = self.initialize_embeddings()
#             self.inputs = tf.keras.Input(shape=(None, state_size), dtype=tf.int32, name='inputs')
#             self.len_state = tf.keras.Input(shape=(), dtype=tf.int32, name='len_state')
            
#             self.input_emb = tf.nn.embedding_lookup(self.all_embeddings['state_embeddings'], self.inputs)

#             if self.model == 'GRU':
#                 gru_out, self.states_hidden = tf.keras.layers.GRU(self.hidden_size, return_sequences=True, return_state=True)(self.input_emb, mask=tf.sequence_mask(self.len_state))
#             elif self.model == 'SASRec':
#                 pos_emb = tf.nn.embedding_lookup(self.all_embeddings['pos_embeddings'],
#                                                   tf.tile(tf.expand_dims(tf.range(tf.shape(self.inputs)[1]), 0), [tf.shape(self.inputs)[0], 1]))
#                 self.seq = self.input_emb + pos_emb
#                 mask = tf.expand_dims(tf.cast(tf.not_equal(self.inputs, item_num), dtype=tf.float32), -1)
#                 self.seq = tf.keras.layers.Dropout(rate=self.dropout_rate)(self.seq, training=self.is_training)
#                 self.seq *= mask
#                 for i in range(self.num_blocks):
#                     self.seq = multihead_attention(queries=tf.math.l2_normalize(self.seq, axis=-1),
#                                                    keys=self.seq,
#                                                    num_units=self.hidden_size,
#                                                    num_heads=self.num_heads,
#                                                    dropout_rate=self.dropout_rate,
#                                                    training=self.is_training,
#                                                    causality=True)
#                     self.seq = feedforward(tf.math.l2_normalize(self.seq, axis=-1), num_units=[self.hidden_size, self.hidden_size], dropout_rate=self.dropout_rate, training=self.is_training)
#                     self.seq *= mask
#                 self.seq = tf.math.l2_normalize(self.seq, axis=-1)
#                 self.states_hidden = tf.gather(self.seq, self.len_state - 1, axis=1)

#             self.output1 = tf.keras.layers.Dense(self.item_num, activation=None)(self.states_hidden)
#             if self.configuration == 'SNQN' or self.configuration == 'SA2C':
#                 self.output2 = tf.keras.layers.Dense(self.item_num, activation=None)(self.states_hidden)
            
#             self.actions = tf.keras.Input(shape=(), dtype=tf.int32, name='actions')
#             self.negative_actions = tf.keras.Input(shape=(self.neg,), dtype=tf.int32, name='negative_actions')
#             self.targetQs_ = tf.keras.Input(shape=(self.item_num,), dtype=tf.float32, name='targetQs_')
#             self.targetQs_selector = tf.keras.Input(shape=(self.item_num,), dtype=tf.float32, name='targetQs_selector')
#             self.reward = tf.keras.Input(shape=(), dtype=tf.float32, name='reward')
#             self.discount = tf.keras.Input(shape=(), dtype=tf.float32, name='discount')
#             self.targetQ_current_ = tf.keras.Input(shape=(self.item_num,), dtype=tf.float32, name='targetQ_current_')
#             self.targetQ_current_selector = tf.keras.Input(shape=(self.item_num,), dtype=tf.float32, name='targetQ_current_selector')
            
#             if self.configuration == 'SA2C':
#                 ce_logits = tf.stop_gradient(self.output2)
#                 target_prob = tf.gather(tf.nn.softmax(ce_logits), self.actions, axis=1)
#                 self.behavior_prob = tf.keras.Input(shape=(), dtype=tf.float32, name='behavior_prob')
#                 self.ips = tf.clip_by_value(tf.divide(target_prob, self.behavior_prob), 0.1, 10)
#                 self.ips = tf.pow(self.ips, self.smooth)
            
#             qloss_positive, _ = trfl.double_qlearning(self.output1, self.actions, self.reward, self.discount, self.targetQs_, self.targetQs_selector)
#             neg_reward = self.reward_negative * tf.ones((self.batch_size,), dtype=tf.float32)
#             qloss_negative = 0
#             for i in range(self.neg):
#                 negative = tf.gather(self.negative_actions, i, axis=1)
#                 qloss_negative += trfl.double_qlearning(self.output1, negative, neg_reward, self.discount, self.targetQ_current_, self.targetQ_current_selector)[0]
#             if self.configuration == 'SNQN':
#                 ce_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.actions, logits=self.output2)
#             elif self.configuration == 'SA2C':
#                 ce_loss_pre = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.actions, logits=self.output2)
#                 ce_loss_post = tf.multiply(self.ips, ce_loss_pre)
#                 q_indexed_positive = tf.stop_gradient(tf.gather(self.output1, self.actions, axis=1))
#                 q_indexed_negative = 0
#                 for i in range(self.neg):
#                     negative = tf.gather(self.negative_actions, i, axis=1)
#                     q_indexed_negative += tf.stop_gradient(tf.gather(self.output1, negative, axis=1))
#                 q_indexed_avg = tf.divide((q_indexed_negative + q_indexed_positive), 1 + self.neg)
#                 advantage = q_indexed_positive - q_indexed_avg
#                 if self.clip >= 0:
#                     advantage = tf.clip_by_value(advantage, self.clip, 10)
#                 ce_loss_post = tf.multiply(advantage, ce_loss_post)
            
#             if self.configuration == 'DQN':
#                 self.loss = tf.reduce_mean(qloss_positive + qloss_negative)
#                 self.opt = tf.keras.optimizers.Adam(self.learning_rate).minimize(self.loss)
#             elif self.configuration == 'SNQN':
#                 self.loss = tf.reduce_mean(qloss_positive + qloss_negative + ce_loss)
#                 self.opt = tf.keras.optimizers.Adam(self.learning_rate).minimize(self.loss)
#             elif self.configuration == 'SA2C':
#                 self.loss1 = tf.reduce_mean(qloss_positive + qloss_negative + ce_loss_pre)
#                 self.opt1 = tf.keras.optimizers.Adam(self.learning_rate).minimize(self.loss1)

#                 self.loss2 = tf.reduce_mean(qloss_positive + qloss_negative + ce_loss_post)
#                 self.opt2 = tf.keras.optimizers.Adam(self.learning_rate2).minimize(self.loss2)
    
#     def initialize_embeddings(self):
#         all_embeddings = {}
#         if not self.pretrain:
#             with self.name_scope:
#                 state_embeddings = tf.Variable(tf.random.normal([self.item_num + 1, self.hidden_size], 0.0, 0.01), name='state_embeddings')
#                 pos_embeddings = tf.Variable(tf.random.normal([self.state_size, self.hidden_size], 0.0, 0.01), name='pos_embeddings')
#                 all_embeddings['state_embeddings'] = state_embeddings
#                 all_embeddings['pos_embeddings'] = pos_embeddings
#         return all_embeddings


        


import tensorflow as tf

class RecQ(tf.keras.Model):
    def __init__(self, item_num, state_size, arg_dict, configuration='SNQN', sequential_model='GRU', pretrain=False):
        super(RecQ, self).__init__()
        
        self.batch_size = 256 if 'batch_size' not in arg_dict else arg_dict['batch_size']
        self.reward_negative = 1 if 'r_negative' not in arg_dict else arg_dict['r_negative']
        self.state_size = state_size
        self.item_num = int(item_num)
        self.learning_rate = 0.001 if 'lr' not in arg_dict else arg_dict['lr']
        self.hidden_size = 64 if 'hidden_factor' not in arg_dict else arg_dict['hidden_factor']
        self.neg = 10 if 'neg' not in arg_dict else arg_dict['neg']
        self.pretrain = pretrain
        self.model = sequential_model
        self.configuration = configuration 
        
        
        # for SASREC
        self.dropout_rate = 0.1
        self.num_blocks = 1
        self.num_heads = 1
        
        # for SA2C
        if self.configuration == 'SA2C':
            self.smooth = 0.0 if 'smooth' not in arg_dict else arg_dict['smooth']
            self.learning_rate2 = 0.001 if 'lr2' not in arg_dict else arg_dict['lr2']
            self.clip = 0.0 if 'clip' not in arg_dict else arg_dict['clip']
        
        self.all_embeddings = self.initialize_embeddings()
        self.gru = tf.keras.layers.GRU(self.hidden_size, return_sequences=True, return_state=True)
        self.q_head = tf.keras.layers.Dense(self.item_num, activation=None)
        if self.configuration == 'SNQN' or self.configuration == 'SA2C':
            self.sv_head = tf.keras.layers.Dense(self.item_num, activation=None)
    
    def initialize_embeddings(self):
        all_embeddings = {}
        if not self.pretrain:
            state_embeddings = tf.random.normal([self.item_num + 1, self.hidden_size], 0.0, 0.01)
            pos_embeddings = tf.random.normal([self.state_size, self.hidden_size], 0.0, 0.01)
            all_embeddings['state_embeddings'] = state_embeddings
            all_embeddings['pos_embeddings'] = pos_embeddings
        return all_embeddings

    def call(self, inputs, len_state):
        input_emb = tf.nn.embedding_lookup(self.all_embeddings['state_embeddings'], inputs)
        #print(input_emb.shape)
        mask=tf.sequence_mask(len_state, maxlen=len(inputs[0]))
        
        expanded_mask = tf.expand_dims(mask, axis=-1)
        expanded_mask = tf.tile(expanded_mask, [1, 1, 64])
        print(expanded_mask.shape)

        #gru_out, self.states_hidden = self.gru(input_emb, mask=expanded_mask)
        gru_out, *h_states = self.gru(input_emb, mask=expanded_mask)

        self.states_hidden = tf.convert_to_tensor(h_states)
        out_q = self.q_head(self.states_hidden)
        out_sv = self.sv_head(self.states_hidden)
        return out_q, out_sv
    
    def my_loss(self, state, len_state, target_Qs, reward, discount, action, target_Qs_selector,
                                        negative, target_Q_current, target_Q_current_selector):
            out_q, out_sv = self(inputs=state, len_state=len_state)
            qloss_positive, _ = trfl.double_qlearning(out_q, action, reward, discount, target_Qs, target_Qs_selector)
            neg_reward = self.reward_negative * tf.ones((self.batch_size,), dtype=tf.float32)
            qloss_negative = 0
            for i in range(self.neg):
                negative_g = tf.gather(negative, i, axis=1)
                qloss_negative += trfl.double_qlearning(out_q, negative_g, neg_reward, discount, target_Q_current, target_Q_current_selector)[0]
            if self.configuration == 'SNQN':
                ce_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=action, logits=out_sv)
                loss = tf.reduce_mean(qloss_positive + qloss_negative + ce_loss)
            return loss


    def compile(self, optimizer):
        super(RecQ, self).compile()
        self.optimizer = optimizer
        self.loss = self.my_loss
                

def train_recq(data_stats, replay_buf, arg_dict):
    state_size = data_stats['state_size'][0]
    item_num = data_stats['item_num'][0]  # total number of items
    reward_click = arg_dict['r_click']
    reward_buy = arg_dict['r_buy']
    QN_1 = RecQ(item_num=item_num, state_size=state_size, arg_dict=arg_dict)
    QN_1.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.005))
    QN_2 = RecQ(item_num=item_num, state_size=state_size, arg_dict=arg_dict)
    QN_2.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.005))
    total_step=0
    num_rows=replay_buf.shape[0]
    num_batches=int(num_rows/arg_dict['batch_size'])
    for i in range(arg_dict['epoch']):
        print(f'$$$$ STARTING EPOCH # {i} $$$$')
        for j in range(num_batches):
            with tf.GradientTape() as tape:
                batch = replay_buf.sample(n=arg_dict['batch_size']).to_dict()
                next_state = list(batch['next_state'].values())
                len_next_state = list(batch['len_next_states'].values())
                # double q learning, pointer is for selecting which network  is target and which is main
                pointer = np.random.randint(0, 2)
                if pointer == 0:
                    mainQN = QN_1
                    targetQN = QN_2
                else:
                    mainQN = QN_2
                    targetQN = QN_2
                main_out_q, main_out_sv = mainQN(inputs=next_state, len_state=len_next_state)
                target_out_q, target_out_sv = targetQN(inputs=next_state, len_state=len_next_state)
                target_Qs = target_out_q
                target_Qs_selector = main_out_q
                # Set target_Qs to 0 for states where episode ends
                is_done = tf.convert_to_tensor(list(batch['is_done'].values()), dtype=tf.bool)
                is_done = tf.expand_dims(is_done, axis=1)
                is_done = tf.tile(is_done, [1, tf.shape(target_Qs)[1]])
                # is_done = tf.tile(tf.convert_to_tensor(list(batch['is_done'].values()), dtype=tf.bool), [1, tf.shape(target_Qs)[1]])
                print(is_done.shape)
                print(target_Qs.shape)
                target_Qs = tf.where(is_done, tf.zeros_like(target_Qs), target_Qs)
                
                state = list(batch['state'].values())
                len_state = list(batch['len_state'].values())
                main_out_q_cur, main_out_sv_cur = mainQN(inputs=state, len_state=len_state)
                target_out_q_cur, target_out_sv_cur = targetQN(inputs=state, len_state=len_state)
                target_Q_current = target_out_q_cur
                target_Q__current_selector = main_out_q_cur
                action = tf.convert_to_tensor(list(batch['action'].values()))
                negative=[]

                for index in range(target_Qs.shape[0]):
                    negative_list=[]
                    for i in range(arg_dict['neg']):
                        neg=np.random.randint(item_num)
                        while neg==action[index]:
                            neg = np.random.randint(item_num)
                        negative_list.append(neg)
                    negative.append(negative_list)

                is_buy=list(batch['is_buy'].values())
                reward=[]
                for k in range(len(is_buy)):
                    reward.append(reward_buy if is_buy[k] == 1 else reward_click)
                reward = tf.convert_to_tensor(reward)
                discount = tf.convert_to_tensor([arg_dict['discount']] * len(action))
                loss_val = mainQN.loss(state, len_state, target_Qs, reward, discount, action, target_Qs_selector,
                                    negative, target_Q_current, target_Q__current_selector)
            gradients = tape.gradient(loss_val, mainQN.trainable_variables)
            mainQN.optimizer.apply_gradients(zip(gradients, mainQN.trainable_variables))
            total_step += 1
            if total_step % 50 == 0:
                print("the loss in %dth batch / %d is: %f" % (total_step, num_batches, loss_val))
    return QN_1, QN_2

# # Example usage:
# # Define hyperparameters
# num_hidden_units = 64
# num_classes = 10
# learning_rate = 0.001
# batch_size = 32
# epochs = 10

# # Create model
# model = SimpleNeuralNetwork(num_hidden_units, num_classes)

# # Compile model
# model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate),
#               loss=tf.keras.losses.SparseCategoricalCrossentropy(),
#               metrics=['accuracy'])

# # Generate some dummy data
# x_train = tf.random.normal(shape=(1000, 20))
# y_train = tf.random.uniform(shape=(1000,), minval=0, maxval=num_classes, dtype=tf.int32)

# # Train the model
# model.fit(x_train, y_train, epochs=epochs, batch_size=batch_size)







import numpy as np

def evaluate(QN_1, eval_ses, state_size, item_num, reward_click, reward_buy, results, pop_dict=None, topk=[5, 10, 15, 20], pickle=False, data_dir='data'):
    if pickle:
        eval_sessions=pd.read_pickle(os.path.join(data_dir, 'sampled_val.df'))
    else:
        eval_sessions = eval_ses
    eval_ids = eval_sessions.session_id.unique()
    groups = eval_sessions.groupby('session_id')
    batch = 100
    evaluated=0
    total_clicks=0.0
    total_purchase = 0.0
    total_reward = [0, 0, 0, 0]
    hit_clicks=[0,0,0,0]
    ndcg_clicks=[0,0,0,0]
    hit_purchase=[0,0,0,0]
    ndcg_purchase=[0,0,0,0]
    
    if QN_1.configuration == 'SA2C':
        off_prob_click=[0.0]
        off_prob_purchase=[0.0]
        off_click_ng=[0.0]
        off_purchase_ng=[0.0]

    while evaluated<len(eval_ids):
        states, len_states, actions, rewards = [], [], [], []
        for i in range(batch):
            if evaluated==len(eval_ids):
                break
            id=eval_ids[evaluated]
            group=groups.get_group(id)
            history=[]
            for index, row in group.iterrows():
                state=list(history)
                len_states.append(state_size if len(state)>=state_size else 1 if len(state)==0 else len(state))
                state=pad_history(state,state_size,item_num)
                states.append(state)
                action=row['item_id']
                is_buy=row['is_buy']
                reward = reward_buy if is_buy == 1 else reward_click
                if is_buy==1:
                    total_purchase+=1.0
                else:
                    total_clicks+=1.0
                actions.append(action)
                rewards.append(reward)
                history.append(row['item_id'])
            evaluated+=1
        if QN_1.configuration == 'DQN':
            prediction = QN_1.output1(states, training=False)
        elif QN_1.configuration == 'SNQN' or QN_1.configuration == 'SA2C':
            prediction = QN_1.output2(states, training=False)
        sorted_list = np.argsort(prediction.numpy())
        calculate_hit(sorted_list,topk,actions,rewards,reward_click,total_reward,hit_clicks,ndcg_clicks,hit_purchase,ndcg_purchase)
        if QN_1.configuration == 'SA2C':
            calculate_off(sorted_list,actions,rewards,reward_click,off_click_ng,off_purchase_ng,off_prob_click,off_prob_purchase,pop_dict)
    print('#############################################################')
    print('total clicks: %d, total purchase:%d' % (total_clicks, total_purchase))
    d = dict()
    for i in range(len(topk)):
        hr_click=hit_clicks[i]/total_clicks
        hr_purchase=hit_purchase[i]/total_purchase
        ng_click=ndcg_clicks[i]/total_clicks
        ng_purchase=ndcg_purchase[i]/total_purchase
        print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
        print('cumulative reward @ %d: %f' % (topk[i],total_reward[i]))
        print('clicks hr ndcg @ %d : %f, %f' % (topk[i],hr_click,ng_click))
        print('purchase hr and ndcg @%d : %f, %f' % (topk[i], hr_purchase, ng_purchase))
        d[topk[i]] = {'reward' : total_reward[i],
                                           'click hr' : hr_click, 'click ndcg' : ng_click,
                                           'purchase hr' : hr_purchase, 'purchase ndcg' : ng_purchase}
    if QN_1.configuration == 'SA2C':
        off_click_ng=off_click_ng[0]/off_prob_click[0]
        off_purchase_ng=off_purchase_ng[0]/off_prob_purchase[0]
        print('off-line corrected evaluation (click_ng,purchase_ng) @10: %f, %f' % (off_click_ng, off_purchase_ng))
        d[10]['off-line click ndcg'] = off_click_ng
        d[10]['off-line purchase ndcg'] = off_purchase_ng
    results.append(d)
    print('#############################################################')
