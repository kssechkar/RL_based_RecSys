import trfl
from trfl import indexing_ops
import tensorflow as tf
from Modules.SASRec import *
from Modules.utils import normalize, extract_axis_1


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