"""
Created on Oct 9, 2019.
Training Actor Critic network.

@author: Monday Ma.
"""


import multiprocessing
import threading
import tensorflow as tf
import numpy as np
import os
import shutil
from Environment import Environment
import matplotlib.pyplot as plt
import random



MAX_STEP = 25  # max step in one epoch 80% people have positive items over 42,90% people have positive item over 25
MAX_EPOCH = 1000   # max epoch number
BATCH_SIZE = 128
# STATE_WINDOW_SIZE = 5
GAMMA = 0.8    # account rate
LR_A = 0.001    # learning rate for actor
LR_C = 0.001    # learning rate for critic
E_GREEDY = 0.1
REPLAY_BUFFER = []

env = Environment()
N_S = env.state_len
N_A = env.action_len


class AC_net(object):
    """
    Build Actor Critic network and training.

    Attributes:
        s: State.
        target_q: Target q_value, for updating Critic network.
        action: Output of Actor.
        q_value: Output of Critic.
        a_params: Actor network parameters.
        c_params: Critic network parameters.
        update_c: Option for updating critic network.
        update_a: Option for updating actor network.
    """
    def __init__(self, scope):
        """
        Build Actor Critic network, define loss\gradient\optimizer

        Args:
            scope: Identifier of actor critic network
        """
        with tf.variable_scope(scope):
            self.s = tf.placeholder(tf.float32, [None, N_S], 'S')
            self.target_q = tf.placeholder(tf.float32, [None, 1], 'target_q')
            self._build_net(scope)

            with tf.name_scope('update_c'):
                reg_set_c = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES, scope=scope + '/critic')
                l2_loss_c = tf.add_n(reg_set_c) / BATCH_SIZE
                td = tf.subtract(self.target_q, self.q_value, name='TD_error')
                self.c_loss = tf.reduce_mean(tf.square(td)) + l2_loss_c
                c_grads = tf.gradients(self.c_loss, self.c_params)
                self.update_c = OPT_C.apply_gradients(zip(c_grads, self.c_params))
            with tf.name_scope('update_a'):
                reg_set_a = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES, scope=scope + '/critic')
                l2_loss_a = tf.add_n(reg_set_a) / BATCH_SIZE
                policy_grads = tf.gradients(ys=self.action, xs=self.a_params,
                                            grad_ys=tf.gradients(-self.q_value + l2_loss_a, self.action))
                self.update_a = OPT_A.apply_gradients(zip(policy_grads, self.a_params))

    def _build_net(self, scope):
        """
        Build the Actor Critic network.

        Args:
            scope: Identifier of AC_net.

        Returns:
            action: Output of Actor network.
            q_value: Output of Critic network.
            a_params: Parameters of Actor network.
            c_params: Parameters of Critic network.
        """
        w_init = tf.random_normal_initializer(0, 0.1)
        l2 = tf.contrib.layers.l2_regularizer(0.01)
        with tf.variable_scope('actor'):
            first_relu = tf.layers.dense(self.s, 200, tf.nn.relu,
                                         kernel_initializer=w_init, kernel_regularizer=l2, name='first_relu')
            second_relu = tf.layers.dense(first_relu, 200, tf.nn.relu,
                                          kernel_initializer=w_init, kernel_regularizer=l2, name='second_relu')
            self.action = tf.layers.dense(second_relu, 100, tf.nn.tanh,
                                     kernel_initializer=w_init, kernel_regularizer=l2, name='action')
        with tf.variable_scope('critic'):
            state_map = tf.layers.dense(self.s, 100, tf.nn.relu,
                                        kernel_initializer=w_init, kernel_regularizer=l2, name='state_map')
            critic_input = tf.concat([self.action, state_map], 1)
            first_relu = tf.layers.dense(critic_input, 100, tf.nn.relu,
                                         kernel_initializer=w_init, kernel_regularizer=l2, name='first_relu')
            second_relu = tf.layers.dense(first_relu, 20, tf.nn.relu,
                                          kernel_initializer=w_init, kernel_regularizer=l2, name='second_relu')
            self.q_value = tf.layers.dense(second_relu, 1,
                                      kernel_initializer=w_init, kernel_regularizer=l2, name='q_value')
        self.a_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=scope + '/actor')
        self.c_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=scope + '/critic')

    def train(self, target_q, s):
        """
        Update AC_net.

        Args:
            target_q: Output of target Critic.
            s: Current state.
        """
        SESS.run([self.update_a, self.update_c], feed_dict={self.s: s,
                                                            self.target_q: target_q})

    def set_params(self, a_params, c_params, t):
        """
        Update target network.

        :param a_params: Train Actor parameters.
        :param c_params: Train Critic parameters.
        :param t: Update weights.
        :return: None
        """
        update_a = [old.assign((1-t)*old + t*new) for old, new in zip(self.a_params, a_params)]
        update_c = [old.assign((1-t)*old + t*new) for old, new in zip(self.c_params, c_params)]

        SESS.run([update_a, update_c])

    def get_action(self, s):
        """
        Get action based on state.

        :param s: State.
        :return: Action.
        """
        s = np.array(s).reshape((1, 300))
        return SESS.run(self.action, feed_dict={self.s: s})

    def get_q(self, s):
        """
        Get q_value based on state.

        :param s: State.
        :return: Q value.
        """
        return SESS.run(self.q_value, feed_dict={self.s: s})


class Memory(object):
    """
    Replay memory saving (St, At, St+1 r).

    Attributes:
        capacity: max records numbers.
        data: numpy array saving records.
        pointer: serial number of current records.
    """
    def __init__(self, capacity, dims):
        """
        Initialize the memory.

        :param capacity: max record numbers.
        :param dims: length of one record.
        """
        self.capacity = capacity
        self.data = np.zeros((capacity, dims))
        self.pointer = 0

    def store_transition(self, s, a, r, s_):
        """
        Store one transition.
        If the memory is full, replace the earlist one

        :param s: current state.
        :param a: action based on current state
        :param r: reward based on current state and action.
        :param s_: next state based on current state and action.
        :return: None
        """
        transition = np.hstack((s, np.squeeze(a), [r], s_))
        index = self.pointer % self.capacity  # replace the old memory with new memory
        self.data[index, :] = transition
        self.pointer += 1

    def sample(self, n):
        """
        Sample a batch for training.

        :param n: Batch size.
        :return: Sampled batch.
        """
        # assert self.pointer >= self.capacity, 'Memory has not been fulfilled'
        indices = np.random.choice(self.capacity, size=n)
        sampled_data = self.data[indices, :]
        b_s = sampled_data[:, :N_S]
        b_a = sampled_data[:, N_S: N_S + N_A]
        b_r = sampled_data[:, -N_S - 1: -N_S]
        b_s_ = sampled_data[:, -N_S:]
        return b_s, b_a, b_r, b_s_


if __name__ == '__main__':
    SESS = tf.Session()
    OPT_A = tf.train.AdamOptimizer(LR_A)
    OPT_C = tf.train.AdamOptimizer(LR_C)
    train_AC = AC_net('train_AC')
    target_AC = AC_net('target_AC')
    saver = tf.train.Saver()
    SESS.run(tf.global_variables_initializer())
    target_AC.set_params(train_AC.a_params, train_AC.c_params, t=0.2)

    memory = Memory(10000, 2*N_S + N_A + 1)

    reward_list = []
    aucs = []
    all_neg_user_num = []
    all_pos_user_num = []
    end_user_num = []
    valid_user_num = []
    end_user = set()
    neg_user = set()
    pos_user = set()

    # For each user, recommendation agent take a few action.
    for step in range(MAX_EPOCH):
        # env.reset()
        reward = 0  # average reward for each epoch.
        # 和每一个用户交互
        for user in range(1, 944):
            if (user not in end_user) and (user not in pos_user) and (user not in neg_user):
                s = env.state[user]
                a = train_AC.get_action(s)
                e = random.random()
                if e < E_GREEDY:
                    noise = np.random.randn(1, N_A)
                    a = np.add(a, noise)
                s_, r, end, info = env.step(user, a)
                if end:
                    end_user.add(user)
                env.set_state(user, s_)
                reward += r
                memory.store_transition(s, a, r, s_)
        # 网络训练和参数更新
        b_s, b_a, b_r, b_s_ = memory.sample(BATCH_SIZE)
        q_ = target_AC.get_q(b_s_)
        target_q = b_r + GAMMA * q_
        train_AC.train(target_q, s=b_s)
        target_AC.set_params(train_AC.a_params, train_AC.c_params, t=0.2)
        # 统计用户的指标
        user_auc = 0
        for user in range(1, 944):
            if user not in end_user:
                items = env._find_positive_history(user)
                s = env.state[user]
                a = train_AC.get_action(s)
                all_label_is, auc = env.auc(user, a)
                if auc >= 0:
                    user_auc += auc
                elif all_label_is == 0:
                    neg_user.add(user)
                else:
                    pos_user.add(user)
        if 943 - len(end_user) - len(neg_user) - len(pos_user) == 0:
            break
        # 记录各项指标
        all_neg_user_num.append(len(neg_user))
        all_pos_user_num.append(len(pos_user))
        end_user_num.append(len(end_user))
        valid_user_num.append(943 - len(end_user) - len(neg_user) - len(pos_user))
        actual_auc = user_auc / (943 - len(end_user) - len(neg_user) - len(pos_user))
        aucs.append(actual_auc)
        average_reward = reward / (943 - len(end_user))
        reward_list.append(average_reward)
        print('step {} finished with average reward {}, auc {}'.format(step, average_reward, actual_auc))
    # 画图
    epoch_num = len(aucs)
    # 奖励图
    plt.subplot(2, 2, 1)
    plt.plot(reward_list)
    plt.xlabel('step')
    plt.ylabel('average reward')
    # auc对比图
    plt.subplot(2, 2, 2)
    plt.plot(aucs, color='r', label='auc')
    plt.plot([0.5839] * epoch_num, color='g', label='auc without reinforcement learning')
    plt.plot([np.mean(aucs)] * epoch_num, color='r', label='average auc', linestyle='-.')
    plt.xlabel('step')
    plt.ylabel('auc')
    plt.legend()
    # 各类用户数目图
    plt.subplot(2, 2, 3)
    plt.plot(end_user_num, color='r', label='end_user_num')
    plt.plot(all_pos_user_num, color='b', label='all_pos_user_num')
    plt.plot(all_neg_user_num, color='b', label='all_neg_user_num', linestyle='-.')
    plt.plot(valid_user_num, color='g', label='valid_user_num')
    plt.xlabel('step')
    plt.ylabel('user_num')
    plt.legend()
    plt.show()
    saver.save(SESS, "/home/mondaym/PycharmProjects/AC_Rec/AC.ckpt")
    SESS.close()
