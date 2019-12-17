import numpy as np
from collections import OrderedDict
import pickle as pkl



ml_100k = './dataset/ml-100k'
FEAT_LEN = 100


class PMF(object):
    def __init__(self, num_feat=FEAT_LEN, epsilon=1, _lambda=0.1, momentum=0.8, maxepoch=10, num_batches=100, batch_size=1000):
        self.num_feat = num_feat  # Number of latent features,
        self.epsilon = epsilon  # learning rate,
        self._lambda = _lambda  # L2 regularization,
        self.momentum = momentum  # momentum of the gradient,
        self.maxepoch = maxepoch  # Number of epoch before stop,
        self.num_batches = num_batches  # Number of batches in each epoch (for SGD optimization),
        self.batch_size = batch_size  # Number of training samples used in each batches (for SGD optimization)

        self.w_Item = None  # Item feature vectors
        self.w_User = None  # User feature vectors

        self.rmse_train = []

    # ***Fit the model with train_tuple and evaluate RMSE on both train and test data.  ***********#
    # ***************** train_vec=TrainData, test_vec=TestData*************#
    def fit(self, train_vec):
        # mean subtraction
        self.mean_inv = np.mean(train_vec[:, 2])  # 评分平均值

        pairs_train = train_vec.shape[0]  # traindata 中条目数

        # 1-p-i, 2-m-c
        num_user = int(np.amax(train_vec[:, 0])) + 1  # 第0列，user总数
        num_item = 1682 + 1  # 第1列，movie总数

        incremental = False  # 增量
        if (not incremental) or (self.w_Item is None):
            # initialize
            self.epoch = 0
            self.w_Item = 0.1 * np.random.randn(num_item, self.num_feat)  # numpy.random.randn 电影 M x D 正态分布矩阵
            self.w_User = 0.1 * np.random.randn(num_user, self.num_feat)  # numpy.random.randn 用户 N x D 正态分布矩阵

            self.w_Item_inc = np.zeros((num_item, self.num_feat))  # 创建电影 M x D 0矩阵
            self.w_User_inc = np.zeros((num_user, self.num_feat))  # 创建用户 N x D 0矩阵

        while self.epoch < self.maxepoch:  # 检查迭代次数
            self.epoch += 1

            # Shuffle training truples
            shuffled_order = np.arange(train_vec.shape[0])  # 根据记录数创建等差array
            np.random.shuffle(shuffled_order)  # 用于将一个列表中的元素打乱

            # Batch update
            for batch in range(self.num_batches):  # 每次迭代要使用的数据量
                # print "epoch %d batch %d" % (self.epoch, batch+1)

                test = np.arange(self.batch_size * batch, self.batch_size * (batch + 1))
                batch_idx = np.mod(test, shuffled_order.shape[0])  # 本次迭代要使用的索引下标

                batch_UserID = np.array(train_vec[shuffled_order[batch_idx], 0], dtype='int32')
                batch_ItemID = np.array(train_vec[shuffled_order[batch_idx], 1], dtype='int32')

                # Compute Objective Function
                pred_out = np.sum(np.multiply(self.w_User[batch_UserID, :],
                                              self.w_Item[batch_ItemID, :]),
                                  axis=1)  # mean_inv subtracted # np.multiply对应位置元素相乘

                rawErr = pred_out - train_vec[shuffled_order[batch_idx], 2] + self.mean_inv

                # Compute gradients
                Ix_User = 2 * np.multiply(rawErr[:, np.newaxis], self.w_Item[batch_ItemID, :]) \
                       + self._lambda * self.w_User[batch_UserID, :]
                Ix_Item = 2 * np.multiply(rawErr[:, np.newaxis], self.w_User[batch_UserID, :]) \
                       + self._lambda * (self.w_Item[batch_ItemID, :])  # np.newaxis :increase the dimension

                dw_Item = np.zeros((num_item, self.num_feat))
                dw_User = np.zeros((num_user, self.num_feat))

                # loop to aggreate the gradients of the same element
                for i in range(self.batch_size):
                    dw_Item[batch_ItemID[i], :] += Ix_Item[i, :]
                    dw_User[batch_UserID[i], :] += Ix_User[i, :]

                # Update with momentum
                self.w_Item_inc = self.momentum * self.w_Item_inc + self.epsilon * dw_Item / self.batch_size
                self.w_User_inc = self.momentum * self.w_User_inc + self.epsilon * dw_User / self.batch_size

                self.w_Item = self.w_Item - self.w_Item_inc
                self.w_User = self.w_User - self.w_User_inc
                # Compute Objective Function after
                if batch == self.num_batches - 1:
                    pred_out = np.sum(np.multiply(self.w_User[np.array(train_vec[:, 0], dtype='int32'), :],
                                                  self.w_Item[np.array(train_vec[:, 1], dtype='int32'), :]),
                                      axis=1)  # mean_inv subtracted
                    rawErr = pred_out - train_vec[:, 2] + self.mean_inv
                    obj = np.linalg.norm(rawErr) ** 2 \
                          + 0.5 * self._lambda * (np.linalg.norm(self.w_User) ** 2 + np.linalg.norm(self.w_Item) ** 2)

                    self.rmse_train.append(np.sqrt(obj / pairs_train))

                    # Print info
                    # if batch == self.num_batches - 1:
                    #     print('Training RMSE: %f' % (self.rmse_train[-1]))

    def set_params(self, parameters):
        if isinstance(parameters, dict):
            self.num_feat = parameters.get("num_feat", 10)
            self.epsilon = parameters.get("epsilon", 1)
            self._lambda = parameters.get("_lambda", 0.1)
            self.momentum = parameters.get("momentum", 0.8)
            self.maxepoch = parameters.get("maxepoch", 20)
            self.num_batches = parameters.get("num_batches", 10)
            self.batch_size = parameters.get("batch_size", 1000)


def load_rating_data(file_path=ml_100k + '/u.data'):
    """
    load movie lens 100k ratings from original rating file.
    need to download and put rating data in /data folder first.
    Source: http://www.grouplens.org/

    :return numpy array formed like [user id, movie id, rating]
    """
    prefer = []
    for line in open(file_path, 'r'):  # 打开指定文件
        (userid, movieid, rating, ts) = line.split('\t')  # 数据集中每行有4项
        uid = int(userid)
        mid = int(movieid)
        rat = float(rating)
        prefer.append([uid, mid, rat])
    data = np.array(prefer)
    return data


def to_reward(rating):
    if rating == 4 or rating == 5:
        return 1
    return -1

def reward_to_label(reward):
    if reward == 1:
        return 1
    else:
        return 0


def load_reward_seq(file_path=ml_100k + '/u.data'):
    sequential_history = dict()  # {user: {item: reward}}
    all_rating = []
    with open(file_path) as f:
        for line in f:
            line = line.rstrip('\n').split('\t')
            line = [eval(item) for item in line]
            all_rating.append(line)
        all_rating.sort(key=lambda x: x[-1])
        for record in all_rating:
            sequential_history.setdefault(record[0], OrderedDict())[record[1]] = to_reward(record[2])
    return sequential_history

# record form: {user{item: rating}}, list form: [user, item, rating]
def record_to_list(data):
    l = []
    for user in data:
        for item in data[user]:
            l.append([user, item, reward_to_label(data[user][item])])
    return l

# 生成器，用来产生tf.data.Dataset
def get_train_data(data):
    """
    将dict形式的输入数据，通过一系列操作变换成输入模型的数据形式。
    :param data: {user: {item: label}}
    :return: [[user_id], [positive_items], [last_item], [label]]
    """
    train_user, train_history, train_item, train_label = [], [], [], []
    for user in data:
        item, reward = data[user].popitem()
        label = 1
        if reward != 1:
            label = 0
        items = [item for item in data[user] if data[user][item] > 0]
        if not items:
            items = [0]
        train_user.append([user])
        train_history.append(items)
        train_item.append([item])
        train_label.append([label])
    return train_user, train_history, train_item, train_label


def get_test_data(data):
    test_user, test_history, test_item, test_label = [], [], [], []
    train_data, test_data = data
    for user in train_data:
        history_items = [item for item in train_data[user] if train_data[user][item] > 0]
        candidates, labels = [], []
        for key, value in test_data[user].items():
            candidates.append(key)
            if value != 1:
                value = 0
            labels.append(value)
        test_user.append([user])
        test_history.append(history_items)
        test_item.append(candidates)
        test_label.append(labels)
    return test_user, test_history, test_item, test_label

def data_split(ratio, record):
    """
    将dict类型的用户反馈数据按照时间和ratio划分成若干份。

    :param ratio: 划分的比例。
    :param record: 最开始的所有用户评分记录。形如 {user: {item: label}}
    :return: 划分后的n份用户评分记录。
    """
    n = len(ratio)
    gates = [sum(ratio[:i+1]) for i in range(n)]
    parts = [{} for _ in range(n)]
    for user in record:
        l = len(record[user])
        i = 0
        part = 0
        for item in record[user]:
            if i / l > gates[part]:
                part += 1
            parts[part].setdefault(user, OrderedDict())[item] = record[user][item]
            i += 1
    return parts


if __name__ == '__main__':
    # pmf = PMF()
    # pmf.fit(load_rating_data(ml_100k + '/u.data'))

    # print(load_rating_seq()[1])
    data = load_reward_seq()
    train_data, test_data1, test_data2 = data_split([0.4, 0.3, 0.3], data)

    np.save('./pmf_train.npy', record_to_list(train_data))

    train_user, train_items, train_item, train_label = get_train_data(train_data)
    test_user1, test_items1, test_item1, test_label1 = get_test_data((train_data, test_data1))
    test_user2, test_items2, test_item2, test_label2 = get_test_data((train_data, test_data2))

    dataset = {}

    dataset['train_user'] = train_user
    dataset['train_items'] = train_items
    dataset['train_item'] = train_item
    dataset['train_label'] = train_label

    dataset['test_user1'] = test_user1
    dataset['test_items1'] = test_items1
    dataset['test_item1'] = test_item1
    dataset['test_label1'] = test_label1

    print(np.shape(test_user1), np.shape(test_items1), np.shape(test_item1), np.shape(test_label1))

    dataset['test_user2'] = test_user2
    dataset['test_items2'] = test_items2
    dataset['test_item2'] = test_item2
    dataset['test_label2'] = test_label2

    with open('./data_3part.pkl', 'wb') as f:
        pkl.dump(dataset, f)









