import tensorflow as tf
import numpy as np
import pandas as pd
import os
import argparse
from collections import deque
from MY.utils import pad_history, calculate_hit, extract_axis_1, calculate_off
#from NextItNetModules import *
from MY.SASRecModules import *

import trfl
from trfl import indexing_ops


class QNetwork:
    def __init__(self, item_num, state_size, arg_dict, configuration='DQN', sequential_model='GRU', pretrain=False, name='DQNetwork'):
        
        
        
        self.batch_size = 256
        if 'batch_size' in arg_dict:
            self.batch_size = arg_dict['batch_size']
        self.reward_negative = 1
        if 'r_negative' in arg_dict:
            self.reward_negative = arg_dict['r_negative']
        
        self.state_size = state_size
        self.item_num = int(item_num)
        
        self.learning_rate = 0.001
        if 'lr' in arg_dict:
            self.learning_rate = arg_dict['lr']
        
        self.hidden_size = 64
        if 'hidden_factor' in arg_dict:
            self.hidden_size = arg_dict['hidden_factor']
        self.neg=10
        if 'neg' in arg_dict:
            self.neg = arg_dict['neg']

        self.pretrain = pretrain

        self.weight=1
        self.model=sequential_model
        self.configuration = configuration 
        
        tf.compat.v1.disable_eager_execution()
        tf.compat.v1.disable_v2_behavior()
        self.is_training = tf.compat.v1.placeholder(tf.bool, shape=())
        # self.save_file = save_file
        self.name = name
        
        # for SASREC
        self.dropout_rate = 0.1
        self.num_blocks = 1
        self.num_heads = 1
        
        # for SA2C
        if self.configuration == 'SA2C':
            self.smooth = 0.0
            if 'smooth' in arg_dict:
                self.smooth = arg_dict['smooth']
            self.learning_rate2 = 0.001
            if 'lr2' in arg_dict:
                self.learning_rate2 = arg_dict['lr2']
            self.clip = 0.0
            if 'clip' in arg_dict:
                self.clip = arg_dict['clip']
        
        with tf.compat.v1.variable_scope(self.name):
            self.all_embeddings=self.initialize_embeddings()
            self.inputs = tf.compat.v1.placeholder(tf.int32, [None, state_size])  # sequence of history, [batchsize,state_size]
            self.len_state = tf.compat.v1.placeholder(tf.int32, [
                None])  # the length of valid positions, because short sesssions need to be padded

            # one_hot_input = tf.one_hot(self.inputs, self.item_num+1)
            self.input_emb = tf.nn.embedding_lookup(self.all_embeddings['state_embeddings'], self.inputs)

            if self.model=='GRU':
                gru_out, self.states_hidden = tf.compat.v1.nn.dynamic_rnn(
                    tf.compat.v1.nn.rnn_cell.GRUCell(self.hidden_size),
                    self.input_emb,
                    dtype=tf.float32,
                    sequence_length=self.len_state,
                )

            if self.model=='SASRec':
                pos_emb = tf.nn.embedding_lookup(self.all_embeddings['pos_embeddings'],
                                                 tf.tile(tf.expand_dims(tf.range(tf.shape(self.inputs)[1]), 0),
                                                         [tf.shape(self.inputs)[0], 1]))
                self.seq = self.input_emb + pos_emb

                mask = tf.expand_dims(tf.cast(tf.not_equal(self.inputs, item_num), dtype=tf.float32), -1)
                # Dropout
                self.seq = tf.compat.v1.layers.dropout(self.seq,
                                             rate=self.dropout_rate,
                                             training=tf.convert_to_tensor(self.is_training))
                self.seq *= mask

                # Build blocks

                for i in range(self.num_blocks):
                    with tf.compat.v1.variable_scope("num_blocks_%d" % i):
                        # Self-attention
                        self.seq = multihead_attention(queries=normalize(self.seq),
                                                       keys=self.seq,
                                                       num_units=self.hidden_size,
                                                       num_heads=self.num_heads,
                                                       dropout_rate=self.dropout_rate,
                                                       is_training=self.is_training,
                                                       causality=True,
                                                       scope="self_attention")

                        # Feed forward
                        self.seq = feedforward(normalize(self.seq), num_units=[self.hidden_size, self.hidden_size],
                                               dropout_rate=self.dropout_rate,
                                               is_training=self.is_training)
                        self.seq *= mask

                self.seq = normalize(self.seq)
                self.states_hidden = extract_axis_1(self.seq, self.len_state - 1)

            self.output1 = tf.compat.v1.layers.dense(self.states_hidden, self.item_num,
                                                            activation=None)
            if self.configuration == 'SNQN' or self.configuration == 'SA2C':
                self.output2= tf.compat.v1.layers.dense(self.states_hidden, self.item_num,
                                                             activation=None)  # all ce logits

            # TRFL way
            self.actions = tf.compat.v1.placeholder(tf.int32, [None])

            self.negative_actions=tf.compat.v1.placeholder(tf.int32,[None,self.neg])

            self.targetQs_ = tf.compat.v1.placeholder(tf.float32, [None, item_num])
            self.targetQs_selector = tf.compat.v1.placeholder(tf.float32, [None,
                                                                 item_num])  # used for select best action for double q learning
            self.reward = tf.compat.v1.placeholder(tf.float32, [None])
            self.discount = tf.compat.v1.placeholder(tf.float32, [None])

            self.targetQ_current_ = tf.compat.v1.placeholder(tf.float32, [None, item_num])
            self.targetQ_current_selector = tf.compat.v1.placeholder(tf.float32, [None,
                                                                 item_num])  # used for select best action for double q learning
            
            if self.configuration == 'SA2C':
                ce_logits = tf.stop_gradient(self.output2)
                target_prob = indexing_ops.batched_index(tf.nn.softmax(ce_logits), self.actions)
                self.behavior_prob = tf.compat.v1.placeholder(tf.float32, [None], name='behavior_prob')
                self.ips = tf.math.divide(target_prob, self.behavior_prob)
                self.ips = tf.clip_by_value(self.ips, 0.1, 10)
                self.ips = tf.pow(self.ips, self.smooth)


            # TRFL double qlearning
            qloss_positive, _ = trfl.double_qlearning(self.output1, self.actions, self.reward, self.discount,
                                                      self.targetQs_, self.targetQs_selector)
            neg_reward=tf.constant(self.reward_negative,dtype=tf.float32, shape=(self.batch_size,))
            qloss_negative=0
            for i in range(self.neg):
                negative=tf.gather(self.negative_actions, i, axis=1)

                qloss_negative+=trfl.double_qlearning(self.output1, negative, neg_reward,
                                                                          self.discount, self.targetQ_current_,
                                                                          self.targetQ_current_selector)[0]
            if self.configuration == 'SNQN':
                ce_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.actions, logits=self.output2)
            elif self.configuration == 'SA2C':
                ce_loss_pre = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.actions, logits=self.output2)
                ce_loss_post = tf.multiply(self.ips,ce_loss_pre)

                q_indexed_positive = tf.stop_gradient(indexing_ops.batched_index(self.output1, self.actions))
                q_indexed_negative = 0
                for i in range(self.neg):
                    negative=tf.gather(self.negative_actions, i, axis=1)
                    q_indexed_negative+=tf.stop_gradient(indexing_ops.batched_index(self.output1, negative))
                q_indexed_avg=tf.divide((q_indexed_negative+q_indexed_positive),1+self.neg)
                advantage=q_indexed_positive-q_indexed_avg

                if self.clip>=0:
                    advantage=tf.clip_by_value(advantage,self.clip,10)

                ce_loss_post = tf.multiply(advantage, ce_loss_post)
            
            if self.configuration == 'DQN':
                self.loss = tf.reduce_mean(qloss_positive+qloss_negative)
                self.opt = tf.compat.v1.train.AdamOptimizer(self.learning_rate).minimize(self.loss)
            elif self.configuration == 'SNQN':
                self.loss = tf.reduce_mean(self.weight*(qloss_positive+qloss_negative)+ce_loss)
                self.opt = tf.compat.v1.train.AdamOptimizer(self.learning_rate).minimize(self.loss)
            elif self.configuration == 'SA2C':
                self.loss1 = tf.reduce_mean(qloss_positive+qloss_negative+ce_loss_pre)
                self.opt1 = tf.compat.v1.train.AdamOptimizer(self.learning_rate).minimize(self.loss1)

                self.loss2 = tf.reduce_mean(self.weight*(qloss_positive + qloss_negative) + ce_loss_post)
                self.opt2 = tf.compat.v1.train.AdamOptimizer(self.learning_rate2).minimize(self.loss2)

    def initialize_embeddings(self):
        all_embeddings = dict()
        if self.pretrain == False:
            with tf.compat.v1.variable_scope(self.name):
                state_embeddings = tf.Variable(tf.random.normal([self.item_num + 1, self.hidden_size], 0.0, 0.01),
                                           name='state_embeddings')
                pos_embeddings = tf.Variable(tf.random.normal([self.state_size, self.hidden_size], 0.0, 0.01),
                                             name='pos_embeddings')
                all_embeddings['state_embeddings'] = state_embeddings
                all_embeddings['pos_embeddings'] = pos_embeddings
        # else:
        #     weight_saver = tf.train.import_meta_graph(self.save_file + '.meta')
        #     pretrain_graph = tf.get_default_graph()
        #     state_embeddings = pretrain_graph.get_tensor_by_name('state_embeddings:0')
        #     with tf.Session() as sess:
        #         weight_saver.restore(sess, self.save_file)
        #         se = sess.run([state_embeddings])[0]
        #     with tf.variable_scope(self.name):
        #         all_embeddings['state_embeddings'] = tf.Variable(se, dtype=tf.float32)
        #     print("load!")
        return all_embeddings

def evaluate(sess, QN_1, eval_ses, state_size, item_num, reward_click, reward_buy, results, pop_dict=None, topk=[5, 10, 15, 20], pickle=False, data_dir='data'):
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
            prediction=sess.run(QN_1.output1, feed_dict={QN_1.inputs: states,QN_1.len_state:len_states,QN_1.is_training:False})
        elif QN_1.configuration == 'SNQN' or QN_1.configuration == 'SA2C':
            prediction=sess.run(QN_1.output2, feed_dict={QN_1.inputs: states,QN_1.len_state:len_states,QN_1.is_training:False})
        sorted_list=np.argsort(prediction)
        calculate_hit(sorted_list,topk,actions,rewards,reward_click,total_reward,hit_clicks,ndcg_clicks,hit_purchase,ndcg_purchase)
        if QN_1.configuration == 'SA2C':
            calculate_off(sorted_list,actions,rewards,reward_click,off_click_ng,off_purchase_ng,off_prob_click,off_prob_purchase,pop_dict)
    print('#############################################################')
    print('total clicks: %d, total purchase:%d' % (total_clicks, total_purchase))
    #print("THE TYPE OF NDCG IS", type(ndcg_clicks), "NDCG:", ndcg_clicks)
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
