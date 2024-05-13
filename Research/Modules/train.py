from Modules.Model import QNetwork, evaluate
from Modules.utils import pad_history, calculate_hit
import pandas as pd
import tensorflow as tf
import os
import numpy as np

def train(data_stats, replay_buf, val_df, arg_dict, results, losses, configuration='DQN', sa2c_switch_step=15000, to_eval=True, pickle=False, data_dir='data'):
    data_directory = data_dir
    if pickle:
        data_statis = pd.read_pickle(
            os.path.join(data_directory, 'data_statis.df'))  # read data statistics, includeing state_size and item_num
    else:
        data_statis = data_stats
    
    state_size = data_statis['state_size'][0]
    item_num = data_statis['item_num'][0]
    reward_click = arg_dict['r_click']
    reward_buy = arg_dict['r_buy']
    reward_negative=arg_dict['r_negative']

    tf.compat.v1.reset_default_graph()

    QN_1 = QNetwork(name='QN_1', state_size=state_size, item_num=item_num, arg_dict=arg_dict, configuration=configuration, sequential_model='SASRec')
    QN_2 = QNetwork(name='QN_2', state_size=state_size, item_num=item_num, arg_dict=arg_dict, configuration=configuration, sequential_model='SASRec')
    if pickle:
        replay_buffer = pd.read_pickle(os.path.join(data_directory, 'replay_buffer.df'))
    else:
        replay_buffer = replay_buf
    
    pop_dict=None
    if configuration=='SA2C':
        with open(os.path.join(data_directory, 'pop_dict.txt'), 'r') as f:
            pop_dict = eval(f.read())

    total_step=0
    sess = tf.compat.v1.Session()
    sess.run(tf.compat.v1.global_variables_initializer())
    if to_eval:
        evaluate(sess, QN_1, val_df, state_size, item_num, reward_click, reward_buy, results, pop_dict=pop_dict, pickle=pickle)
    num_rows=replay_buffer.shape[0]
    num_batches=int(num_rows/arg_dict['batch_size'])
    for i in range(arg_dict['epoch']):
        print(f'$$$$ STARTING EPOCH # {i} $$$$')
        for j in range(num_batches):
            batch = replay_buffer.sample(n=arg_dict['batch_size']).to_dict()
            next_state = list(batch['next_state'].values())
            len_next_state = list(batch['len_next_states'].values())
            
            pointer = np.random.randint(0, 2)
            if pointer == 0:
                mainQN = QN_1
                target_QN = QN_2
            else:
                mainQN = QN_2
                target_QN = QN_1
            target_Qs = sess.run(target_QN.output1,
                                    feed_dict={target_QN.inputs: next_state,
                                            target_QN.len_state: len_next_state,
                                            target_QN.is_training:True})
            target_Qs_selector = sess.run(mainQN.output1,
                                            feed_dict={mainQN.inputs: next_state,
                                                        mainQN.len_state: len_next_state,
                                                        mainQN.is_training:True})

            is_done = list(batch['is_done'].values())
            for index in range(target_Qs.shape[0]):
                if is_done[index]:
                    target_Qs[index] = np.zeros([item_num])

            state = list(batch['state'].values())
            len_state = list(batch['len_state'].values())
            target_Q_current = sess.run(target_QN.output1,
                                        feed_dict={target_QN.inputs: state,
                                                    target_QN.len_state: len_state,
                                                    target_QN.is_training:True})
            target_Q__current_selector = sess.run(mainQN.output1,
                                                    feed_dict={mainQN.inputs: state,
                                                                mainQN.len_state: len_state,
                                                                mainQN.is_training:True})
            action = list(batch['action'].values())
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
            discount = [arg_dict['discount']] * len(action)
                
            if configuration == 'DQN' or configuration == 'SNQN':
                loss, _ = sess.run([mainQN.loss, mainQN.opt],
                                feed_dict={mainQN.inputs: state,
                                            mainQN.len_state: len_state,
                                            mainQN.targetQs_: target_Qs,
                                            mainQN.reward: reward,
                                            mainQN.discount: discount,
                                            mainQN.actions: action,
                                            mainQN.targetQs_selector: target_Qs_selector,
                                            mainQN.negative_actions:negative,
                                            mainQN.targetQ_current_:target_Q_current,
                                            mainQN.targetQ_current_selector:target_Q__current_selector,
                                            mainQN.is_training:True
                                            })
            elif configuration == 'SA2C':
                if total_step < sa2c_switch_step:
                    loss, _ = sess.run([mainQN.loss1, mainQN.opt1],
                                    feed_dict={mainQN.inputs: state,
                                                mainQN.len_state: len_state,
                                                mainQN.targetQs_: target_Qs,
                                                mainQN.reward: reward,
                                                mainQN.discount: discount,
                                                mainQN.actions: action,
                                                mainQN.targetQs_selector: target_Qs_selector,
                                                mainQN.negative_actions: negative,
                                                mainQN.targetQ_current_: target_Q_current,
                                                mainQN.targetQ_current_selector: target_Q__current_selector,
                                                mainQN.is_training:True
                                                })
                else:
                    behavior_prob = []
                    for a in action:
                        behavior_prob.append(pop_dict[a])

                    loss, _ = sess.run([mainQN.loss2, mainQN.opt2],
                                    feed_dict={mainQN.inputs: state,
                                                mainQN.len_state: len_state,
                                                mainQN.targetQs_: target_Qs,
                                                mainQN.reward: reward,
                                                mainQN.discount: discount,
                                                mainQN.actions: action,
                                                mainQN.targetQs_selector: target_Qs_selector,
                                                mainQN.negative_actions: negative,
                                                mainQN.targetQ_current_: target_Q_current,
                                                mainQN.targetQ_current_selector: target_Q__current_selector,
                                                mainQN.behavior_prob: behavior_prob,
                                                mainQN.is_training:True
                                                })
            total_step += 1
            if total_step % 50 == 0:
                print("the loss in %dth batch is: %f" % (total_step, loss))
                losses.append(loss)
            if to_eval and (total_step % 250 == 0):
                evaluate(sess, QN_1, val_df, state_size, item_num, reward_click, reward_buy, results, pop_dict=pop_dict, pickle=pickle)
    return QN_1, sess




def test(sess, QN_1, data_stats, test_df, results, r_click=0.2, r_buy=1, pickle=False, data_dir='data'):
    data_directory = data_dir
    if pickle:
        data_statis = pd.read_pickle(
            os.path.join(data_directory, 'data_statis.df'))
    else:
        data_statis = data_stats
    
    state_size = data_statis['state_size'][0]  # the length of history to define the state
    item_num = data_statis['item_num'][0]  # total number of items
    reward_click = r_click
    reward_buy = r_buy
    
    pop_dict=None
    with open(os.path.join(data_directory, 'pop_dict.txt'), 'r') as f:
        pop_dict = eval(f.read())
    
    evaluate(sess, QN_1, test_df, state_size, item_num, reward_click, reward_buy, results, pop_dict=pop_dict, pickle=pickle)
    sess.close()
