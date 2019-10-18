"""
Created on Oct 9, 2019

Creating a environment in recommendation systems.

@author: Monday Ma
"""

from ML100k_processing import PMF
from ML100k_processing import load_rating_data, load_rating_seq
from sklearn.metrics import roc_auc_score
import numpy as np
import random

ml_100k = '/home/mondaym/PycharmProjects/ARLMR/dataset/ml-100k'


class Environment(object):
    """
    Environment(one user) of recommendation Agent.
    """
    def __init__(self):
        self.state_len = 300
        self.action_len = 100
        self.state_window_size = 5
        self.user_id = 0
        self.bad_user = []
        pmf = PMF()
        pmf.fit(load_rating_data(ml_100k + '/u.data'))
        print('training complete...')
        self.user_features = pmf.w_User
        self.item_features = pmf.w_Item
        self.user_record = load_rating_seq()  # {user: {item, reward}}
        self.user_history = {k: v for k, v in zip(range(1, 944), self._init_positive_history())}


    def _init_positive_history(self):
        """
        Find the initial N(self.state_window_size) positive items as initial history of each user.
        If positive item in the record less than N, raise RuntimeError.

        :return: Initial positive items of this user.
        """
        for user in range(1, 944):
            history = []
            for item in self.user_record[user]:
                if self.user_record[user][item] == 1:
                    history.append(item)
                if len(history) == self.state_window_size:
                    yield history
                    break
            if len(history) < 5:
                self.bad_user.append(user)
                yield history
        return None

    @staticmethod
    def _find_latest_positive_history(record, history):
        """
        :param record: Record of current user.
        :param history: History of current user.
        :return: latest N positive history.
        """
        positive_history = [item for item in history if record[item] == 1]
        if len(positive_history) <= 5:
            return positive_history
        return [item for item in history if record[item] == 1][-5:]

    @staticmethod
    def _find_candidate(record, history):
        """
        :param record: Record of current user.
        :param history: History of current user.
        :return: Candidates for current user.
        """
        return [item for item in record if item not in history]

    def state_encode(self, user, items):
        """
        Given user and history items, encode it to state.

        :param user: User id.
        :param items: item id.
        :return: Current state.
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

    @staticmethod
    def get_reward(record, item):
        """
        Compute reward based on recommended item.
        :param item: Recommended item.
        :return: Reward value.
        """
        return record[item]

    def reset(self):
        """
        Initial state.

        Compute each user's state, candidates, and positive items

        Random choose a user.
        Select the initial n positive items as initial state and remove from candidates.

        :return: Initial state
        """
        # random choose a user, and return the initial state.
        self.user_id = random.choice(range(1, 944))
        while self.user_id in self.bad_user:
            self.user_id = random.choice(range(1, 944))
        items = self.user_history[self.user_id]
        self.state = self.state_encode(self.user_id, items)
        return self.state

    def step(self, action):
        """
        Take an action, get into a new state, get a reward.

        :param action: Action.
        :return: Current state, reward, end flag, information.
        """
        # recommend an item from user record.
        candidates = self._find_candidate(self.user_record[self.user_id], self.user_history[self.user_id])
        candidate_mat = [self.item_features[i] for i in candidates]
        ratings = list(np.matmul(candidate_mat, np.transpose(action)))
        recommend_item = candidates[ratings.index(max(ratings))]
        reward = self.get_reward(self.user_record[self.user_id], recommend_item)
        if reward > 0:
            self.user_history[self.user_id].append(recommend_item)
            info = 'recommendation success'
        else:
            info = 'recommendation fail'
        pos_items = self._find_latest_positive_history(self.user_record[self.user_id], self.user_history[self.user_id])
        state = self.state_encode(self.user_id, pos_items)
        end = not bool(candidates)
        return state, reward, end, info

    def auc(self, user, action):
        candidate_mat = [self.item_features[k] for k in self.user_record[user].keys()]
        labels = [1 if v > 0 else 0 for v in self.user_record[user].values()]
        ratings = list(np.matmul(candidate_mat, np.transpose(action)))
        p = list(map(lambda x: 1 / (1 + np.e ** (-x)), ratings))
        return roc_auc_score(labels, p)


if __name__ == '__main__':
    env = Environment()
    # s0 = env.reset()
    # s, r, end, info = env.step(env.user_features[env.user_id])
