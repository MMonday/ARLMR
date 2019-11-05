import matplotlib.pyplot as plt
import numpy as np

# replot
MAX_EPOCH = 100
with open('one_user_per_epoch.txt', 'r') as f:
    epochs = eval(f.readline())
    reward_list = eval(f.readline())
    auc = eval(f.readline())
    plt.subplot(1, 2, 1)
    plt.plot(np.arange(MAX_EPOCH), reward_list)
    plt.xlabel('epoch')
    plt.ylabel('reward')
    plt.subplot(1, 2, 2)
    plt.plot(np.arange(MAX_EPOCH), auc)
    plt.xlabel('epoch')
    plt.ylabel('auc')
    plt.show()


