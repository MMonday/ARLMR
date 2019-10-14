"""
Created on Oct 9, 2019

Creating a environment in recommendation systems.

@author: Monday Ma
"""

from ML100k_processing import PMF
from ML100k_processing import load_rating_data, load_rating_seq
import numpy as np
import random

ml_100k = '/home/mondaym/PycharmProjects/ARLMR/dataset/ml-100k'

class Environment(object):
    """
    Environment(one user) of recommendation Agent.

    Attributes:
        state_len: Length of state vector.
        action_len: Lenth of action vector.
        state_window_size: history like numbers.
        user_id: Current user id.
        items: History items.
        user_features: All user features.
        item_features: All item features.
        user_history: Dictionary in form {user: (item, reward)}
        candidate: Candidate items.
        state: Current state.
    """
    def __init__(self):
        self.state_len = 300
        self.action_len = 100
        self.state_window_size = 5
        self.user_id = 0
        self.items = []
        pmf = PMF()
        pmf.fit(load_rating_data(ml_100k + '/u.data'))
        print('training complete...')
        self.user_features = pmf.w_User
        self.item_features = pmf.w_Item
        self.user_history = load_rating_seq()
        self.candidate = []
        self.state = None

    def state_encode(self, user, items):
        """
        Given user and history items, encode it to state.

        :param user: User id.
        :param items: item id.
        :return: Current state.
        """
        item_features = [self.item_features[item] for item in items]
        item_ave = np.mean(item_features, axis=0)
        state = np.concatenate([self.user_features[user],
                                np.multiply(self.user_features[user], item_ave),
                                item_ave], axis=0)
        return state

    def get_reward(self, item):
        """
        Compute reward based on recommended item.
        :param item: Recommended item.
        :return: Reward value.
        """
        user_history = self.user_history[self.user_id]
        user_items = [i[0] for i in user_history]
        user_ratings = [j[1] for j in user_history]
        if item in user_items:
            reward = user_ratings[user_items.index(item)]
        else:
            reward = 0
        return reward

    def reset(self):
        """
        Initial state.

        Random choose a user.
        Select the initial n positive items as initial state and remove from candidates.

        :return: Initial state
        """
        self.user_id = random.choice(range(1, 944))
        user_record = self.user_history[self.user_id]
        self.candidate = [i[0] for i in user_record]
        positive_items = [i[0] for i in user_record if i[1] > 0]
        self.items = positive_items[0: self.state_window_size]
        for item in self.items:
            self.candidate.remove(item)
        self.state = self.state_encode(self.user_id, self.items)
        return self.state

    def step(self, action):
        """
        Take an action, get into a new state, get a reward.

        :param action: Action.
        :return: Current state, reward, end flag, information.
        """
        candidate_mat = [self.item_features[i] for i in self.candidate]
        ratings = list(np.matmul(candidate_mat, np.transpose(action)))
        recommend_item = self.candidate[ratings.index(max(ratings))]
        reward = self.get_reward(recommend_item)
        if reward > 0:
            self.items.remove(self.items[0])
            self.items.append(recommend_item)
            info = 'recommendation success'
        else:
            info = 'recommendation fail'
        self.state = self.state_encode(self.user_id, self.items)
        self.candidate.remove(recommend_item)
        end = not bool(self.candidate)
        return self.state, reward, end, info


if __name__ == '__main__':
    env = Environment()
    s0 = env.reset()
    s, r, end, info = env.step(env.user_features[env.user_id])
