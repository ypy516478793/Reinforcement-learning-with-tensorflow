"""
This part of code is the Deep Q Network (DQN) brain.

view the tensorboard picture about this DQN structure on: https://morvanzhou.github.io/tutorials/machine-learning/reinforcement-learning/4-3-DQN3/#modification

View more on my tutorial page: https://morvanzhou.github.io/tutorials/

Using:
Tensorflow: r1.2
"""

import numpy as np
import tensorflow as tf
from collections import defaultdict

np.random.seed(1)
tf.set_random_seed(1)


# Deep Q Network off-policy
class DeepQNetwork:
    def __init__(
            self,
            n_actions,
            n_features,
            n_goals=1,
            learning_rate=0.01,
            reward_decay=0.9,
            e_greedy=0.1,
            e_decrement=0.01,
            replace_target_iter=200,
            memory_size=2000,
            batch_size=32,
            meta=False,
            output_graph=False,
    ):
        self.n_actions = n_actions
        self.n_features = n_features
        self.n_goals = n_goals
        self.n_new_features = n_features if n_goals == 1 else n_features + 1
        self.lr = learning_rate
        self.gamma = reward_decay
        self.epsilon = np.ones(n_goals) * e_greedy
        self.replace_target_iter = replace_target_iter
        self.memory_size = memory_size
        self.batch_size = batch_size
        self.de_anneal = e_decrement
        self.memory_counter = 0
        self.goal_attempts = defaultdict(int)
        self.goal_success = defaultdict(int)
        self.meta = meta

        # total learning step
        self.learn_step_counter = 0

        # initialize zero memory [s, a, r, s_] or [{s, g}, a, r, {s_, g}]
        if n_goals == 1:
            self.memory = np.zeros((self.memory_size, n_features * 2 + 2))
        else:
            self.memory = np.zeros((self.memory_size, (n_features + 1) * 2 + 2))

        # consist of [target_net, evaluate_net]
        scope_type = 'meta_controller' if meta else 'controller'
        self._build_net(scope_type)

        t_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=scope_type + '/target_net')
        e_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=scope_type + '/eval_net')

        with tf.variable_scope(scope_type + '/soft_replacement'):
            self.target_replace_op = [tf.assign(t, e) for t, e in zip(t_params, e_params)]

        self.sess = tf.Session()

        if output_graph:
            # $ tensorboard --logdir=logs
            tf.summary.FileWriter("logs/", self.sess.graph)

        self.sess.run(tf.global_variables_initializer())
        self.cost_his = []

    def _build_net(self, scope_type):
        # ------------------ all inputs ------------------------
        self.s = tf.placeholder(tf.float32, [None, self.n_new_features], name='s')  # input State
        self.s_ = tf.placeholder(tf.float32, [None, self.n_new_features], name='s_')  # input Next State
        self.r = tf.placeholder(tf.float32, [None, ], name='r')  # input Reward
        self.a = tf.placeholder(tf.int32, [None, ], name='a')  # input Action

        w_initializer, b_initializer = tf.random_normal_initializer(0., 0.3), tf.constant_initializer(0.1)

        with tf.variable_scope(scope_type):
            # ------------------ build evaluate_net ------------------
            with tf.variable_scope('eval_net'):
                e1 = tf.layers.dense(self.s, 20, tf.nn.relu, kernel_initializer=w_initializer,
                                     bias_initializer=b_initializer, name='e1')
                self.q_eval = tf.layers.dense(e1, self.n_actions, kernel_initializer=w_initializer,
                                              bias_initializer=b_initializer, name='q')

            # ------------------ build target_net ------------------
            with tf.variable_scope('target_net'):
                t1 = tf.layers.dense(self.s_, 20, tf.nn.relu, kernel_initializer=w_initializer,
                                     bias_initializer=b_initializer, name='t1')
                self.q_next = tf.layers.dense(t1, self.n_actions, kernel_initializer=w_initializer,
                                              bias_initializer=b_initializer, name='t2')

            with tf.variable_scope('q_target'):
                q_target = self.r + self.gamma * tf.reduce_max(self.q_next, axis=1, name='Qmax_s_')    # shape=(None, )
                self.q_target = tf.stop_gradient(q_target)
            with tf.variable_scope('q_eval'):
                a_indices = tf.stack([tf.range(tf.shape(self.a)[0], dtype=tf.int32), self.a], axis=1)
                self.q_eval_wrt_a = tf.gather_nd(params=self.q_eval, indices=a_indices)    # shape=(None, )
            with tf.variable_scope('loss'):
                self.loss = tf.reduce_mean(tf.squared_difference(self.q_target, self.q_eval_wrt_a, name='TD_error'))
            with tf.variable_scope('train'):
                self._train_op = tf.train.RMSPropOptimizer(self.lr).minimize(self.loss)

    def store_transition(self, s, a, r, s_, goal=None):
        # if not hasattr(self, 'memory_counter'):
        #     self.memory_counter = 0
        if self.n_goals == 1:
            transition = np.hstack((s, a, r, s_))
        else:
            transition = np.hstack((s, goal, a, r, s_, goal))
        # replace the old memory with new memory
        index = self.memory_counter % self.memory_size
        self.memory[index, :] = transition
        self.memory_counter += 1

    def choose_action(self, observation, goal=None):
        # to have batch dimension when feed into tf placeholder
        if self.n_goals == 1:
            observation = observation[np.newaxis, :]
            epsilon_threshold = self.epsilon
        else:
            observation = np.append(observation, goal)[np.newaxis, :]
            epsilon_threshold = self.epsilon[goal]

        if np.random.uniform() > epsilon_threshold:
            # forward feed the observation and get q value for every actions
            actions_value = self.sess.run(self.q_eval, feed_dict={self.s: observation})
            action = np.argmax(actions_value)
        else:
            action = np.random.randint(0, self.n_actions)
        return action

    def learn(self):
        # check to replace target parameters
        if self.learn_step_counter % self.replace_target_iter == 0:
            self.sess.run(self.target_replace_op)
            # print('\ntarget_params_replaced\n')

        # sample batch memory from all memory
        if self.memory_counter > self.memory_size:
            sample_index = np.random.choice(self.memory_size, size=self.batch_size)
        else:
            sample_index = np.random.choice(self.memory_counter, size=self.batch_size)
        batch_memory = self.memory[sample_index, :]

        _, cost = self.sess.run(
            [self._train_op, self.loss],
            feed_dict={
                self.s: batch_memory[:, :self.n_new_features],
                self.a: batch_memory[:, self.n_features],
                self.r: batch_memory[:, self.n_features + 1],
                self.s_: batch_memory[:, -self.n_new_features:],
            })

        self.cost_his.append(cost)

    def criticize(self, next_state, goal):
        if next_state == goal:
            in_reward = 1
            goal_reached = True
        else:
            in_reward = 0
            goal_reached = False
        return in_reward, goal_reached

    def anneal(self, steps=1, goal=None, adaptively=False, success=False):
        epsilon_threshold = self.epsilon if self.n_goals == 1 else self.epsilon[goal]
        if epsilon_threshold > 0.1:
            if adaptively:
                success_rate = self.goal_success[goal] / self.goal_attempts[goal]
                if success_rate == 0:
                    epsilon_threshold -= self.de_anneal * steps
                else:
                    epsilon_threshold = 1 - success_rate
            else:
                epsilon_threshold -= self.de_anneal * steps

            if success:
                epsilon_threshold -= self.de_anneal * 20

            if epsilon_threshold < 0.1:
                epsilon_threshold = 0.1
        if self.n_goals == 1:
            self.epsilon = epsilon_threshold
        else:
            self.epsilon[goal] = epsilon_threshold


if __name__ == '__main__':
    DQN = DeepQNetwork(3,4, output_graph=True)