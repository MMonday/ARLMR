from ML100k_processing import PMF, load_rating_data
import tensorflow as tf
import pickle as pkl
import random
import time
from sklearn.metrics import roc_auc_score
import numpy as np
# from tensorflow.python import debug as dbg
import matplotlib.pyplot as plt

ML_100K = './dataset/ml-100k'

class BaseModel(object):
    def __init__(self):
        pmf = PMF()
        pmf.fit(np.load('./pmf_train.npy'))
        print('pretraining complete...')
        self.user_features = pmf.w_User
        self.dim = len(self.user_features[0])
        self.user_features[0] = np.array([0.0 for _ in range(self.dim)])
        self.item_features = pmf.w_Item
        self.item_features[0] = np.array([0.0 for _ in range(self.dim)])
        self.lr = 0.0001
        self.max_epoch = 100
        self.batch_size = 32
        self.load_data()
        self.create_graph()

    def load_data(self):
        with open('./data_3part.pkl', 'rb') as f:
            dataset = pkl.load(f)
        self.train_user = dataset['train_user']
        self.train_items = dataset['train_items']
        self.train_item = dataset['train_item']
        self.train_label = dataset['train_label']

        self.test_user1 = dataset['test_user1']
        self.test_items1 = dataset['test_items1']
        self.test_item1 = dataset['test_item1']
        self.test_label1 = dataset['test_label1']

        self.test_user2 = dataset['test_user2']
        self.test_items2 = dataset['test_items2']
        self.test_item2 = dataset['test_item2']
        self.test_label2 = dataset['test_label2']

    def create_graph(self):
        w_init = tf.random_normal_initializer(0, 0.1)
        l2 = tf.contrib.layers.l2_regularizer(0.01)

        self.user_id = tf.placeholder(tf.int32, [None, 1])
        self.history_id = tf.placeholder(tf.int32, [None, None])
        self.item_id = tf.placeholder(tf.int32, [None, 1])
        self.label = tf.placeholder(tf.float32, [None, 1])

        self.user_feature = tf.squeeze(tf.nn.embedding_lookup(self.user_features, self.user_id), axis=1)
        self.history_feature = tf.nn.embedding_lookup(self.item_features, self.history_id)
        self.item_feature = tf.squeeze(tf.nn.embedding_lookup(self.item_features, self.item_id), axis=1)

        self.history_ave = tf.reduce_mean(self.history_feature, axis=1)

        self.user_exp = tf.concat([self.user_feature,
                                  tf.multiply(self.user_feature, self.history_ave),
                                  self.history_ave], axis=1)  # b, 3e

        first_relu = tf.layers.dense(self.user_exp, 200, tf.nn.relu,
                                     kernel_initializer=w_init, kernel_regularizer=l2, name='first_relu')
        second_relu = tf.layers.dense(first_relu, 200, tf.nn.relu,
                                      kernel_initializer=w_init, kernel_regularizer=l2, name='second_relu')
        self.user = tf.layers.dense(second_relu, 100, tf.nn.tanh,
                                    kernel_initializer=w_init, kernel_regularizer=l2, name='action')  # b, e
        self.output = tf.nn.sigmoid(tf.diag_part(tf.matmul(self.user, tf.transpose(self.item_feature))))  # b, 1

        self.loss = tf.losses.log_loss(self.label, tf.reshape(self.output, [-1, 1]))
        self.opt = tf.train.AdamOptimizer(self.lr).minimize(self.loss)

    @staticmethod
    def pad(batch):
        # 将一个batch内的历史记录电影个数补齐。
        max_len = max((len(i) for i in batch))
        padded_batch = list(map(lambda x: x + [0] * (max_len - len(x)), batch))
        return padded_batch

    def batch_gen(self, user, history, item, label):
        total_num = len(user)
        index = 0
        while index < total_num - 1 - self.batch_size:
            user_batch = [user[index] for index in range(index, index + self.batch_size)]
            history_batch = [history[index] for index in range(index, index + self.batch_size)]
            item_batch = [item[index] for index in range(index, index + self.batch_size)]
            label_batch = [label[index] for index in range(index, index + self.batch_size)]
            index += self.batch_size
            yield user_batch, self.pad(history_batch), item_batch, label_batch
        user_batch = [user[index] for index in range(index, total_num)]
        history_batch = [history[index] for index in range(index, total_num)]
        item_batch = [item[index] for index in range(index, total_num)]
        label_batch = [label[index] for index in range(index, total_num)]
        yield user_batch, self.pad(history_batch), item_batch, label_batch

    def auc(self, test_user, test_history, test_item, test_label):
        sum_auc = 0
        for user, history, item, label in zip(test_user, test_history, test_item, test_label):
            test_num = len(label)
            predictions = self.sess.run(self.output, feed_dict={self.user_id: [user] * test_num,
                                                                self.history_id: [history] * test_num,
                                                                self.item_id: np.reshape(item, [-1, 1]),
                                                                self.label: np.reshape(label, [-1, 1])})
            try:
                this_auc = roc_auc_score(label, predictions)
            except ValueError:
                this_auc = 0
            sum_auc += this_auc
        return sum_auc / len(test_user)

    def train(self):
        with tf.Session() as self.sess:
            self.sess.run(tf.global_variables_initializer())
            print('initial:')
            print('train_loss:', self.sess.run(self.loss, feed_dict={self.user_id: self.train_user,
                                                                     self.history_id: self.pad(self.train_items),
                                                                     self.item_id: self.train_item,
                                                                     self.label: self.train_label}))
            print('test_auc1:', self.auc(self.test_user1, self.test_items1, self.test_item1, self.test_label1))
            print('test_auc2:', self.auc(self.test_user2, self.test_items2, self.test_item2, self.test_label2))
            loss, aucs1, aucs2 = [], [], []
            for epoch in range(self.max_epoch):
                # 打乱训练数据顺序
                train_data = list(zip(self.train_user, self.train_items, self.train_item, self.train_label))
                random.shuffle(train_data)
                train_user, train_items, train_item, train_label = zip(*train_data)
                batch = self.batch_gen(train_user, train_items, train_item, train_label)
                # 训练
                try:
                    while 1:
                        user_batch, history_batch, item_batch, label_batch = next(batch)
                        self.sess.run(self.opt, feed_dict={self.user_id: user_batch,
                                                           self.history_id: history_batch,
                                                           self.item_id: item_batch,
                                                           self.label: label_batch})
                except StopIteration:
                    train_loss = self.sess.run(self.loss, feed_dict={self.user_id: self.train_user,
                                                                     self.history_id: self.pad(self.train_items),
                                                                     self.item_id: self.train_item,
                                                                     self.label: self.train_label})
                    test_auc1 = self.auc(self.test_user1, self.test_items1, self.test_item1, self.test_label1)
                    test_auc2 = self.auc(self.test_user2, self.test_items2, self.test_item2, self.test_label2)
                    loss.append(train_loss)
                    aucs1.append(test_auc1)
                    aucs2.append(test_auc2)
                    if (len(aucs1) > 3) and (aucs1[-1] < aucs1[-2] < aucs1[-3]):
                        print('early stop.')
                        break
                    print('train_loss:{}    test_auc1:{}    test_auc2:{}'.format(train_loss, test_auc1, test_auc2))
        return loss, aucs1, aucs2

if __name__ == '__main__':
    m = BaseModel()
    loss, aucs1, aucs2 = m.train()
    epochs = len(loss)
    plt.subplot(2, 2, 1)
    plt.plot(np.arange(epochs), loss)
    plt.xlabel('epoch')
    plt.ylabel('train_loss')
    plt.subplot(2, 2, 3)
    plt.plot(np.arange(epochs), aucs1)
    plt.xlabel('epoch')
    plt.ylabel('valid_auc')
    plt.subplot(2, 2, 4)
    plt.plot(np.arange(epochs), aucs2)
    plt.xlabel('epoch')
    plt.ylabel('test_auc')
    plt.show()


