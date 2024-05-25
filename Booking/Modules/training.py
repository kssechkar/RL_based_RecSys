import pandas as pd
import os
import tensorflow as tf
from Modules.Network import QNetwork
import numpy as np
from Modules.utils import pad_history


def train_no_eval(data_stats, replay_buf, arg_dict, losses, configuration='DQN', sa2c_switch_step=10000, pickle=False, data_dir='data/processed'):
    # Network parameters
    data_directory = data_dir
    if pickle:
        data_statis = pd.read_pickle(
            os.path.join(data_directory, 'data_statis.df'))  # read data statistics, includeing state_size and item_num
    else:
        data_statis = data_stats

    state_size = data_statis['state_size'][0]  # the length of history to define the state
    item_num = data_statis['item_num'][0]  # total number of items
    reward_click = arg_dict['r_click']
    reward_buy = arg_dict['r_buy']
    reward_negative=arg_dict['r_negative']
    topk=[5,10,15,20]
    # save_file = 'pretrain-GRU/%d' % (hidden_size)

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
    # Initialize variables
    sess.run(tf.compat.v1.global_variables_initializer())
    num_rows=replay_buffer.shape[0]
    num_batches=int(num_rows/arg_dict['batch_size'])
    for i in range(arg_dict['epoch']):
        print(f'$$$$$$$ STARTING EPOCH {i} $$$$$$$')
        for j in range(num_batches):
            batch = replay_buffer.sample(n=arg_dict['batch_size']).to_dict()

                #state = list(batch['state'].values())

            next_state = list(batch['next_state'].values())
            len_next_state = list(batch['len_next_states'].values())
            # double q learning, pointer is for selecting which network  is target and which is main
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
            # Set target_Qs to 0 for states where episode ends
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
                print("the loss in %dth batch / %d is: %f" % (total_step, num_batches, loss))
                losses.append(loss)
    return QN_1, sess

def predict_booking_batched(sess, QN_1, eval_ses, data_statis, close=True):
    cl = ['utrip_id'] + [f'city_id_{i+1}' for i in range(10)]
    predictions = pd.DataFrame(columns=cl)
    state_size = data_statis['state_size'][0]  # the length of history to define the state
    item_num = data_statis['item_num'][0]  # total number of items
    eval_sessions = eval_ses
    eval_ids = eval_sessions.session_id.unique()
    groups = eval_sessions.groupby('session_id')
    batch = 512
    #saver = tf.compat.v1.train.Saver()
    #with tf.compat.v1.Session() as sess:
    #saver.restore(sess, 'current_booking_mdl.ckpt')
    evaluated=0
    while evaluated<len(eval_ids):
        batch_ids = np.array(eval_ids[evaluated:evaluated+batch]).reshape(-1, 1)
        states, len_states = [], []
        for i in range(batch):
            if evaluated==len(eval_ids):
                break
            id=eval_ids[evaluated]
            group = groups.get_group(id)
            state = pad_history(group['item_id'].to_list(), state_size, item_num)
            len_state = len(state)
            if len(state) >= state_size:
                len_state = state_size
            elif len(state) == 0:
                len_state = 1
            states.append(state)
            len_states.append(len_state)
            evaluated += 1
        if QN_1.configuration == 'DQN':
            prediction = sess.run(QN_1.output1, feed_dict={QN_1.inputs: states, QN_1.len_state: len_states, QN_1.is_training: False})
        elif QN_1.configuration == 'SNQN' or QN_1.configuration == 'SA2C':
            prediction = sess.run(QN_1.output2, feed_dict={QN_1.inputs: states, QN_1.len_state: len_states, QN_1.is_training: False})
        sorted_list = np.argsort(prediction, axis=1)[:, -10:][:, ::-1]
        to_add = pd.DataFrame(np.concatenate((batch_ids, sorted_list), axis=1), columns=predictions.columns)
        predictions = pd.concat([predictions, to_add], ignore_index=True)
    if close:
      sess.close()
    return predictions
