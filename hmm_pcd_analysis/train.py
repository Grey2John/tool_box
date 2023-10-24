import os
import numpy as np
from hmmlearn import hmm


down_points_dir = {}
down_txt_path = 'F:\earth_rosbag\\test_hmm\data\down_sample_set_bag5'
file_list = os.listdir(down_txt_path)
for l in file_list:
    file = os.path.join(down_txt_path, l)
    # print("process {}".format(file))
    down_points = []
    with open(file, 'r') as f:
        lines = f.readlines()
        for line in lines:
            one_list = line.split(', ')
            one_point = {}
            one_point['xyz'] = [float(str_num) for str_num in one_list[0:3]]
            # one_point.append(len(one_list)-4)
            one_point['obs']=[]
            for str in one_list[4:-1]:
                one_point['obs'].append(int(str))
            one_point['obs'].append(int(one_list[-1][0]))
            down_points.append(one_point)
    down_points_dir[len(one_list)-4] = down_points

print('length is {}'.format(down_points_dir[10][1]))

# initial configure lambda
# states = ["none", "stone", "sand"]
# vocabulary = [0,1,2,3,4,5,6]  # obs state
# # data 每种观测次数要统计到状态列表中 [ 20  73   0   7   0   0   0]
# train_x_list=[]
# for l in down_points_dir[100]:
#     one_p = [l['obs'].count(i) for i in vocabulary]
#     train_x_list.append(one_p)
# train_x = np.array(train_x_list, dtype=int)
# print('train data length is {}'.format(train_x.shape))
# print(train_x[:2, :])
# # initial
# startprob = np.array([0.3, 0.6, 0.1])  # pi
# transmat = np.array([[0.8, 0.1, 0.1],
#                     [0.1, 0.8, 0.1],
#                     [0.1, 0.1, 0.8]])
# emissionprob = np.array([[0.76, 0.01, 0.01, 0.1, 0.1, 0.01, 0.01],
#                       [0.01, 0.76, 0.01, 0.1, 0.1, 0.01, 0.01],
#                       [0.01, 0.01, 0.76, 0.1, 0.1, 0.01, 0.01]])
# model = hmm.MultinomialHMM(n_components=len(states),
#                             n_trials=100,  # 观测序列的次数
#                             n_iter=10,
#                             tol=0.001,
#                             init_params='ste')
#
# model.n_features = len(vocabulary)
# model.startprob_ = startprob
# model.transmat_ = transmat
# model.emissionprob_ = emissionprob
#
# model.fit( train_x, train_x.shape[0] )  # 版本数量
#
# print(model.transmat_)
# print("初始状态概率：", np.around(model.startprob_, decimals=4))
# print("状态转移概率：", np.around(model.transmat_, decimals=4))
# print("发射矩阵：", np.around(model.emissionprob_, decimals=4))