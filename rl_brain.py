import numpy as np
import tensorflow as tf

# np.random.seed(1)
# tf.set_random_seed(1)


class DoubleDQN:
    def __init__(self,
                 n_actions,
                 n_features,
                 learning_rate=0.05,
                 reward_decay=0.9,
                 e_greedy=0.0,
                 replace_target_iter=200,
                 memory_size=1000,
                 batch_size=128,
                 e_greedy_inc=0.0005
                 ):
        self.n_actions = n_actions
        self.n_features = n_features
        self.lr = learning_rate
        self.gamma = reward_decay
        self.epsilon = e_greedy
        self.replace_target_iter = replace_target_iter
        self.memory_size = memory_size
        self.batch_size = batch_size
        self.epsilon_inc = e_greedy_inc

        self.learn_step = 0
        self.memory_count = 0
        self.memory = np.zeros((self.memory_size, n_features * 2 + 2))
        self._build_net()

        t_params = tf.get_collection('target_net_params')
        e_params = tf.get_collection('eval_net_params')
        self.replace_target_op = [tf.assign(t, e) for t, e in zip(t_params, e_params)]

        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())

    def _build_net(self):
        def build_layers(s, c_names, n_l1, w_initializer, b_initializer):
            with tf.variable_scope('l1'):
                w1 = tf.get_variable('w1', [self.n_features, n_l1], initializer=w_initializer, collections=c_names)
                b1 = tf.get_variable('b1', [1, n_l1], initializer=b_initializer, collections=c_names)
                l1 = tf.nn.relu(tf.matmul(s, w1) + b1)

            with tf.variable_scope('l2'):
                w2 = tf.get_variable('w2', [n_l1, self.n_actions], initializer=w_initializer, collections=c_names)
                b1 = tf.get_variable('b2', [1, self.n_actions], initializer=w_initializer, collections=c_names)
                out = tf.matmul(l1, w2) + b1
            return out

        self.s = tf.placeholder(tf.float32, [None, self.n_features], name='s')
        self.q_target = tf.placeholder(tf.float32, [None, self.n_actions], name='Q_target')

        with tf.variable_scope('eval_net'):
            c_names, n_l1, w_initializer, b_initializer = \
                ['eval_net_params', tf.GraphKeys.GLOBAL_VARIABLES], 150, \
                tf.random_normal_initializer(0., 0.3), tf.constant_initializer(0.1)
            self.q_evel = build_layers(self.s, c_names, n_l1, w_initializer, b_initializer)

        with tf.variable_scope('loss'):
            self.loss = tf.reduce_mean(tf.squared_difference(self.q_target, self.q_evel))

        with tf.variable_scope('train'):
            self._train_op = tf.train.RMSPropOptimizer(self.lr).minimize(self.loss)

        self.s_ = tf.placeholder(tf.float32, [None, self.n_features], name='s_')
        with tf.variable_scope('target_net'):
            c_names = ['target_net_params', tf.GraphKeys.GLOBAL_VARIABLES]
            self.q_next = build_layers(self.s_, c_names, n_l1, w_initializer, b_initializer)

    def store_transition(self, s, a, r, s_):
        transition = np.hstack((s, [a, r], s_))
        index = self.memory_count % self.memory_size
        self.memory[index, :] = transition
        self.memory_count += 1

    def choose_action(self, observation):
        observation = np.array(observation)
        observation_ = observation.reshape(1, 96)
        action_value = self.sess.run(self.q_evel, feed_dict={self.s: observation_})
        parking_num = np.argmax(action_value)

        parking_number = len(observation) - 2
        if observation[parking_num] != 0:
            tmp_n1 = parking_num
            while tmp_n1 >= 0:
                if observation[tmp_n1] == 0:
                    break
                tmp_n1 -= 1

            tmp_n2 = parking_num
            while tmp_n2 < parking_number - 1:
                if observation[tmp_n2] == 0:
                    break
                tmp_n2 += 1

            # if tmp_n2 == parking_number and tmp_n1 == -1:
            #     return -1

            if tmp_n1 == -1:
                parking_num = tmp_n2
            elif tmp_n2 == parking_number:
                parking_num = tmp_n1
            else:
                if abs(tmp_n2 - parking_num) < abs(tmp_n1 - parking_num):
                    parking_num = tmp_n2
                else:
                    parking_num = tmp_n1

        if np.random.uniform() > self.epsilon:
            index_list = list()
            for i in range(parking_number):
                if observation[i] == 0:
                    index_list.append(i)
            parking_num = np.random.choice(index_list)
        return parking_num

    def learn(self):
        if self.learn_step % self.replace_target_iter == 0:
            self.sess.run(self.replace_target_op)
            # print('target params replaced\n')

        if self.memory_count > self.memory_size:
            sample_index = np.random.choice(self.memory_size, size=self.batch_size)
        else:
            sample_index = np.random.choice(self.memory_count, size=self.batch_size)

        batch_memory = self.memory[sample_index, :]

        q_next, q_eval_next = self.sess.run(
            [self.q_next, self.q_evel],
            feed_dict={self.s_: batch_memory[:, -self.n_features:],
                       self.s: batch_memory[:, -self.n_features:]}
        )

        q_eval = self.sess.run(self.q_evel, {self.s: batch_memory[:, :self.n_features]})

        q_target = q_eval.copy()

        batch_index = np.arange(self.batch_size, dtype=np.int32)
        eval_act_index = batch_memory[:, self.n_features].astype(int)

        reward = batch_memory[:, self.n_features + 1]

        max_act_next = np.argmax(q_eval_next, axis=1)
        selected_q_next = q_next[batch_index, max_act_next]

        q_target[batch_index, eval_act_index] = reward + self.gamma * selected_q_next

        _, self.cost = self.sess.run([self._train_op, self.loss],
                                     feed_dict={self.s: batch_memory[:, :self.n_features],
                                     self.q_target: q_target})

        self.epsilon = min(self.epsilon + self.epsilon_inc, 1)
        self.learn_step += 1


if __name__ == '__main__':
    print(round(0.99999 * 304))