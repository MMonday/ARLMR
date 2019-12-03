import matplotlib.pyplot as plt
import numpy as np
import pickle as pkl
import random
import pandas as pd
from ML100k_processing import load_rating_data

# replot
# MAX_EPOCH = 100
# with open('one_user_per_epoch.txt', 'r') as f:
#     epochs = eval(f.readline())
#     reward_list = eval(f.readline())
#     auc = eval(f.readline())
#     plt.subplot(1, 2, 1)
#     plt.plot(np.arange(MAX_EPOCH), reward_list)
#     plt.xlabel('epoch')
#     plt.ylabel('reward')
#     plt.subplot(1, 2, 2)
#     plt.plot(np.arange(MAX_EPOCH), auc)
#     plt.xlabel('epoch')
#     plt.ylabel('auc')
#     plt.show()


# with open('./data_3part.pkl', 'rb') as f:
#     dataset = pkl.load(f)
# print(dataset['train_user'])
# print(dataset['train_items'])
# print(dataset['train_item'])
# print(dataset['train_label'])

# data = [i for i in range(100)]
# def my_iter(data, size):
#     index = 0
#     while index < len(data) - 1 - size:
#         res = [data[index] for index in range(index, index + size)]
#         index += size
#         yield res
#     res = [data[index] for index in range(index, len(data))]
#     yield res
#
# a = my_iter(data, 5)
# while 1:
#     print(next(a))

a = [1, 2, 3]
plt.plot(a, color='r', label='red')
plt.plot([3] * 3, color='g', label='green')
plt.plot([np.mean(a)] * 3, color='r', label='average', linestyle='-.')
plt.xlabel('epoch')
plt.ylabel('auc')
plt.legend()
plt.show()