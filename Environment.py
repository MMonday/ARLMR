"""
Created on Oct 9, 2019

Creating a environment in recommendation systems.

@author: Monday Ma
"""

from ML100k_processing import PMF
from ML100k_processing import load_rating_data, load_reward_seq, data_split
from sklearn.metrics import roc_auc_score
from heapq import nlargest
import numpy as np
import random

ml_100k = '/home/mondaym/PycharmProjects/ARLMR/dataset/ml-100k'


class Environment(object):
    """
    给推荐系统提供状态和动作状态转移关系。
    状态编码： S = f(用户表示， 历史最近n个正项目) = 用户表示 + 用户表示 × 历史最近5个正项目平均 + 历史最近5个正项目平均
    P(s_t+1 | s_t, a_t)： 候选电影与输出动作内积得到评分，进而排序取第一个推荐结果，得到用户的反馈，若为正项目，加到历史中，状态变化，否则状态不变。
    """
    def __init__(self):
        self.state_len = 300
        self.action_len = 100
        self.state_window_size = 5
        self.user_id = 0
        self.bad_user = []   # 没有正评价的消极用户
        pmf = PMF()
        pmf.fit(np.load('./pmf_train.npy'))
        print('training complete...')
        self.user_features = pmf.w_User  # 预训练用户表示， 100维
        self.item_features = pmf.w_Item  # 预训练电影表示， 100维
        self.item_num = len(self.item_features) - 1
        self.user_num = len(self.user_features) - 1
        self.top_n = 1
        self.user_record = load_reward_seq()  # 用户日志： 形如{user: {item, reward}}
        # 包括被推荐的历史和交互的历史
        self.user_history = {k: v for k, v in self._init_history()}  # 开始训练时每个用户的历史记录，形如{user: [movie_ids]}
        self.state = {}
        for user in range(1, 944):
            self.set_state(user, self.state_encode(user, self._find_positive_history(user)))

    def _init_history(self):
        """
        选取用户日志中的最初n个正项目，初始化每个用户的历史。

        :yield: 每个用户的初始历史
        """
        train_data, _ = data_split([0.4, 0.6], self.user_record)
        for user in range(1, 944):
            yield user, list(train_data[user].keys())

    def _find_positive_history(self, user_id):
        """
        :param record: 用户日志。class OrderedDict: {item: reward}.
        :param history: 当前用户此刻的所有历史。
        :return: 当前用户此刻的最近5个正历史，如果不足的话，把所有的正历史都算上。
        """
        record, history = self.user_record[user_id], self.user_history[user_id]
        positive_history = [item for item in history if item in record and record[item] == 1]
        return positive_history

    def _find_candidate(self, user):
        """
        :param user: 当前用户id。
        :return: 候选电影 = 所有电影 - 已被推荐过的电影
        """
        return [item for item in range(1, self.item_num + 1) if item not in self.user_history[user]]

    def rated_items(self, user, items):
        return [item for item in items if item in self.user_record[user]]

    def state_encode(self, user, items):
        """
        S = f(用户表示， 历史最近n个正项目) = 用户表示 + 用户表示 × 历史最近5个正项目平均 + 历史最近5个正项目平均

        :param user: 用户id。
        :param items: 电影 id。
        :return: 当前状态编码。
        """
        if not items:
            item_features = [[0.0 for _ in range(self.action_len)]]
        else:
            item_features = [self.item_features[item] for item in items]
        item_ave = np.mean(item_features, axis=0)
        state = np.concatenate([self.user_features[user],
                                np.multiply(self.user_features[user], item_ave),
                                item_ave], axis=0)
        return state

    def set_state(self, user, s):
        self.state[user] = s

    def get_reward(self, user, items):
        """
        :param user: 当前用户。
        :param item: 被推荐的电影。
        :return: 得到的奖励。
        """
        total_reward = 0
        for item in items:
            if item in self.user_record[user]:
                total_reward += self.user_record[user][item]
        return total_reward

    def reset(self):
        """
        随机选取一个用户作为训练用户。
        初始化所有用户的历史。

        :return: 被选择用户的初始状态编码。
        """
        # random choose a user, and return the initial state.
        self.user_id = random.choice(range(1, 944))
        while self.user_id in self.bad_user:
            self.user_id = random.choice(range(1, 944))
        self.user_history = {k: v for k, v in self._init_history()}
        items = self.user_history[self.user_id]
        s_ = self.state_encode(self.user_id, items)
        return s_

    def step(self, user, action):
        """
        对当前用户做出推荐，用户到达新状态，并得到奖励。

        :param action: 动作向量，根据此做出推荐。
        :return: 下一步状态, 奖励, 是否结束, 日志记录.
        """
        end = False
        candidates = self._find_candidate(user)
        candidate_mat = [self.item_features[i] for i in candidates]
        ratings = np.matmul(candidate_mat, np.transpose(action)).reshape(-1).tolist()
        recommend_items = nlargest(self.top_n, candidates, key=lambda x: ratings[candidates.index(x)])
        reward = self.get_reward(user, recommend_items)
        self.user_history[user] += recommend_items
        if reward > 0:
            info = 'recommendation success'
        else:
            info = 'recommendation fail'
        pos_items = self._find_positive_history(user)
        state = self.state_encode(user, pos_items)
        if len(candidates) == 1:
            end = True
        return state, reward, end, info

    def auc(self, user, action):
        """
        计算如果对一个user做出action，该user的auc是多少。
        :param user: 用户id。
        :param action: 对该用户做出的动作。
        :return: 该用户的auc。
        """
        candidate = self.rated_items(user, self._find_candidate(user))
        if len(candidate) <= 1:
            return 0, -1
        candidate_mat = [self.item_features[i] for i in candidate]
        labels = [1 if self.user_record[user][i] > 0 else 0 for i in candidate]
        ratings = np.matmul(candidate_mat, np.transpose(action)).reshape(-1).tolist()
        all_label_is = -1
        try:
            auc = roc_auc_score(labels, ratings)
        except ValueError:
            auc = -1
            if 0 not in labels:
                all_label_is = 1
            if 1 not in labels:
                all_label_is = 0
        return all_label_is, auc


if __name__ == '__main__':
    env = Environment()
    # s0 = env.reset()
    # s, r, end, info = env.step(env.user_features[env.user_id])
