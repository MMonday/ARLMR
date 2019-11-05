from ML100k_processing import PMF, load_rating_data, parse_train, parse_test
import tensorflow as tf
from sklearn.metrics import roc_auc_score
import numpy as np
# from tensorflow.python import debug as dbg

ML_100K = './dataset/ml-100k'

class BaseModel(object):
    def __init__(self):
        pmf = PMF()
        pmf.fit(load_rating_data(ML_100K + '/u.data'))
        print('pretraining complete...')
        self.user_features = pmf.w_User
        self.item_features = pmf.w_Item
        self.dim = len(self.user_features[0])
        self.lr = 0.01
        self.max_epoch = 100
        self.train_batch_gen()
        self.test_batch_gen()
        self.create_graph()

    def train_batch_gen(self):
        filenames = ['train.tfr']
        raw_dataset = tf.data.TFRecordDataset(filenames=filenames)
        parsed_dataset = raw_dataset.map(parse_train)
        batches = parsed_dataset.shuffle(1000).padded_batch(32, padded_shapes=([], [None], [], []))
        self.train_batch_iterator = batches.make_initializable_iterator()
        self.user_id, self.history, self.item_id, self.label = self.train_batch_iterator.get_next()

    def test_batch_gen(self):
        filenames = ['test.tfr']
        raw_dataset = tf.data.TFRecordDataset(filenames=filenames)
        parsed_dataset = raw_dataset.map(parse_test)
        batches = parsed_dataset.batch(1)
        self.test_batch_iterator = batches.make_initializable_iterator()
        self.test_user_id, self.test_history, self.test_item_id, self.test_label = self.test_batch_iterator.get_next()

    def create_graph(self):
        w_init = tf.random_normal_initializer(0, 0.1)
        l2 = tf.contrib.layers.l2_regularizer(0.01)

        # train part
        item_ave = tf.reduce_mean(tf.nn.embedding_lookup(self.item_features, self.history), axis=1)  # b, e
        user_features = tf.nn.embedding_lookup(self.user_features, self.user_id)  # b, e
        item_feature = tf.nn.embedding_lookup(self.item_features, self.item_id) # b, e

        self.user_exp = tf.concat([user_features,
                                  tf.multiply(user_features, item_ave),
                                  item_ave], axis=1)  # b, 3e

        first_relu = tf.layers.dense(self.user_exp, 200, tf.nn.relu,
                                     kernel_initializer=w_init, kernel_regularizer=l2, name='first_relu')
        second_relu = tf.layers.dense(first_relu, 200, tf.nn.relu,
                                      kernel_initializer=w_init, kernel_regularizer=l2, name='second_relu')
        self.user = tf.layers.dense(second_relu, 100, tf.nn.tanh,
                                    kernel_initializer=w_init, kernel_regularizer=l2, name='action')  # b, e
        self.output = tf.nn.sigmoid(tf.diag_part(tf.matmul(self.user, tf.transpose(item_feature))))  # b, 1

        self.loss = tf.losses.log_loss(self.label, self.output)
        self.opt = tf.train.AdamOptimizer(self.lr).minimize(self.loss)

    def train(self):
        with tf.Session() as self.sess:
            # self.sess = dbg.LocalCLIDebugWrapperSession(self.sess, ui_type='readline')
            self.sess.run(tf.global_variables_initializer())
            for epoch in range(self.max_epoch):
                self.sess.run(self.train_batch_iterator.initializer)
                try:
                    while True:
                        self.sess.run(self.opt)
                        print('batch')
                except:
                    pass
                finally:
                    print('epoch')


if __name__ == '__main__':
    m = BaseModel()
    m.train()


