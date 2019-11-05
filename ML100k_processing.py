import numpy as np
from collections import OrderedDict
import tensorflow as tf
import IPython.display as display



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
        num_item = int(np.amax(train_vec[:, 1])) + 1  # 第1列，movie总数

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
    if rating == 3:
        return 0
    return -1


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


# 生成器，用来产生tf.data.Dataset
def get_train_data(data):
    """
    将dict形式的输入数据，通过一系列操作变换成输入模型的数据形式。
    :param data: {user: {item: label}}
    :return: [[user_id], [positive_items], [last_item], [label]]
    """
    for user in data:
        item, reward = data[user].popitem()
        label = 1
        if reward != 1:
            label = 0
        items = [item for item in data[user] if data[user][item] > 0]
        if not items:
            items = [0]
        yield user, items, item, label


def get_test_data(data):
    train_data, test_data = data
    for user in train_data:
        history_items = [item for item in train_data[user] if train_data[user][item] > 0]
        candidates, labels = [], []
        for key, value in test_data[user].items():
            candidates.append(key)
            if value != 1:
                value = 0
            labels.append(value)
        yield user, history_items, candidates, labels



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


def to_TFRecord(data, name, filename):
    """
    把训练数据和测试数据转换成TFRecord文件的形式。
    :param data: 如果是训练数据，则只需要训练集的data;如果是测试数据，那么是（训练数据，测试数据）
    :param name: 标示是生成训练数据还是测试数据。
    :return: 什么都不返回，在文件根目录下留下一个TFRecord的文件。
    """
    if name == 'train':
        it = get_train_data
    elif name == 'test':
        it = get_test_data
    else:
        raise ValueError('name参数是train或者test')
    with tf.python_io.TFRecordWriter(filename) as writer:
        for user_id, positive_items, last_item, label in it(data):
            features = {}
            features['user_id'] = tf.train.Feature(int64_list=tf.train.Int64List(value=[user_id]))
            features['positive_items'] = tf.train.Feature(int64_list=tf.train.Int64List(value=positive_items))
            if name == 'train':
                features['last_item'] = tf.train.Feature(int64_list=tf.train.Int64List(value=[last_item]))
                features['label'] = tf.train.Feature(int64_list=tf.train.Int64List(value=[label]))
            elif name == 'test':
                features['candidates'] = tf.train.Feature(int64_list=tf.train.Int64List(value=last_item))
                features['labels'] = tf.train.Feature(int64_list=tf.train.Int64List(value=label))
            example = tf.train.Example(features=tf.train.Features(feature=features))
            writer.write(example.SerializeToString())


def parse_train(serial):
    feature_description = {
        'user_id': tf.FixedLenFeature(dtype=tf.int64, shape=[]),
        'positive_items': tf.VarLenFeature(dtype=tf.int64),
        'last_item': tf.FixedLenFeature(dtype=tf.int64, shape=[]),
        'label': tf.FixedLenFeature(dtype=tf.int64, shape=[])
    }
    feats = tf.parse_single_example(serial, feature_description)
    user_id = feats['user_id']
    history_items = tf.sparse_tensor_to_dense(feats['positive_items'])
    item = feats['last_item']
    label = feats['label']
    return user_id, history_items, item, label


def parse_test(serial):
    feature_description = {
        'user_id': tf.FixedLenFeature(dtype=tf.int64, shape=[]),
        'positive_items': tf.VarLenFeature(dtype=tf.int64),
        'candidates': tf.VarLenFeature(dtype=tf.int64),
        'labels': tf.VarLenFeature(dtype=tf.int64)
    }
    feats = tf.parse_single_example(serial, feature_description)
    user_id = feats['user_id']
    history_items = tf.sparse_tensor_to_dense(feats['positive_items'])
    item = tf.sparse_tensor_to_dense(feats['candidates'])
    label = tf.sparse_tensor_to_dense(feats['labels'])
    return user_id, history_items, item, label


if __name__ == '__main__':
    # pmf = PMF()
    # pmf.fit(load_rating_data(ml_100k + '/u.data'))

    # print(load_rating_seq()[1])

    # 允许tensor不在图内跑
    tf.enable_eager_execution()

    # 生成训练时的TFRecord
    # data = load_reward_seq()
    # train_data, test_data = data_split([0.7, 0.3], data)
    # to_TFRecord(data, name='train', filename='train.tfr')
    # to_TFRecord((train_data, test_data), name='test', filename='test.tfr')

    # 解析训练时的TFRecord
    filenames = ['test.tfr']
    raw_dataset = tf.data.TFRecordDataset(filenames=filenames)
    parsed_dataset = raw_dataset.map(parse_test)
    batch = parsed_dataset.shuffle(100).padded_batch(1, padded_shapes=([], [None], [None], [None]))
    for item in batch:
        print(np.shape(item[0]))
        print(np.shape(item[1]))
        print(np.shape(item[2]))
        print(np.shape(item[3]))









