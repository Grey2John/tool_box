import os
import numpy as np
import data_loader
from sklearn import hmm

start_pron = np.array([0.3, 0.6, 0.1]).T  # pi
trans_state = np.array([[0.8, 0.1, 0.1],
                    [0.1, 0.8, 0.1],
                    [0.1, 0.1, 0.8]])
emission = np.array([[0.76, 0.01, 0.01, 0.1, 0.1, 0.01, 0.01],
                      [0.01, 0.76, 0.01, 0.1, 0.1, 0.01, 0.01],
                      [0.01, 0.01, 0.76, 0.1, 0.1, 0.01, 0.01]])
states = ["none", "stone", "sand"]
vocabulary = [0,1,2,3,4,5,6]  # obs state


class HMM:
    """
    https://applenob.github.io/machine_learning/HMM/
    ----------
    A : numpy.ndarray
        State transition probability matrix
    B: numpy.ndarray
        Output emission probability matrix with shape(N, number of output types)
    pi: numpy.ndarray
        Initial state probablity vector
    """
    def __init__(self, A, B, pi):
        self.A = A
        self.B = B
        self.pi = pi

    def forward(self, obs_seq):
        """前向算法"""
        N = self.A.shape[0]
        T = len(obs_seq)

        # 整个序列的概率 F
        F = np.zeros((N, T))
        F[:, 0] = self.pi * self.B[:, obs_seq[0]] # initial prob, 3x1 * 3x1 = 3x1

        for t in range(1, T): # recursion prob
            for n in range(N):
                F[n, t] = np.dot(F[:, t - 1], (self.A[:, n])) * self.B[n, obs_seq[t]]  # sum(alpha * a)*b

        return F

    def backward(self, obs_seq):
        """后向算法"""
        N = self.A.shape[0]
        T = len(obs_seq)
        # X保存后向概率矩阵
        X = np.zeros((N, T))
        X[:, -1:] = 1

        for t in reversed(range(T - 1)):
            for n in range(N):
                X[n, t] = np.sum(X[:, t + 1] * self.A[n, :] * self.B[:, obs_seq[t + 1]])

        return X

    def baum_welch_train(self, observations, criterion=0.05):
        """无监督学习算法——Baum-Weich算法"""
        n_states = self.A.shape[0]
        n_samples = len(observations)  # 单个序列

        done = False
        while not done:
            # alpha_t(i) = P(O_1 O_2 ... O_t, q_t = S_i | hmm)
            # Initialize alpha
            alpha = self.forward(observations)

            # beta_t(i) = P(O_t+1 O_t+2 ... O_T | q_t = S_i , hmm)
            # Initialize beta
            beta = self.backward(observations)
            # ξ_t(i,j)=P(i_t=q_i,i_{i+1}=q_j|O,λ)
            xi = np.zeros((n_states, n_states, n_samples - 1))
            for t in range(n_samples - 1):
                denom = np.dot(np.dot(alpha[:, t].T, self.A) * self.B[:, observations[t + 1]].T, beta[:, t + 1])
                for i in range(n_states):
                    numer = alpha[i, t] * self.A[i, :] * self.B[:, observations[t + 1]].T * beta[:, t + 1].T
                    xi[i, :, t] = numer / denom

            # γ_t(i)：gamma_t(i) = P(q_t = S_i | O, hmm)
            gamma = np.sum(xi, axis=1)
            # Need final gamma element for new B
            # xi的第三维长度n_samples-1，少一个，所以gamma要计算最后一个
            prod = (alpha[:, n_samples - 1] * beta[:, n_samples - 1]).reshape((-1, 1))
            gamma = np.hstack((gamma, prod / np.sum(prod)))  # append one more to gamma!!!

            # 更新模型参数
            newpi = gamma[:, 0]
            newA = np.sum(xi, 2) / np.sum(gamma[:, :-1], axis=1).reshape((-1, 1))
            newB = np.copy(self.B)
            num_levels = self.B.shape[1]
            sumgamma = np.sum(gamma, axis=1)
            for lev in range(num_levels):
                mask = observations == lev
                newB[:, lev] = np.sum(gamma[:, mask], axis=1) / sumgamma

            # 检查是否满足阈值
            if np.max(abs(self.pi - newpi)) < criterion and \
                    np.max(abs(self.A - newA)) < criterion and \
                    np.max(abs(self.B - newB)) < criterion:
                done = 1
            self.A[:], self.B[:], self.pi[:] = newA, newB, newpi
        return newA, newB, newpi


if __name__ == "__main__":
    model = HMM(trans_state, emission, start_pron)
    data = data_loader.PointDataLoader("/home/zlh/data/sandpile_source/data/test3/r3live_4pixel/sample_bag0")
    data.down_sample_loader()

    a, b, p = model.baum_welch_train(data.down_points_dir[100][1]['obs'])
    print(a)
    print(b)
    print(p)
# initial configure lambda

# # data 每种观测次数要统计到状态列表中 [ 20  73   0   7   0   0   0]
# train_x_list=[]
# for l in down_points_dir[100]:
#     one_p = [l['obs'].count(i) for i in vocabulary]
#     train_x_list.append(one_p)
# train_x = np.array(train_x_list, dtype=int)
# print('train data length is {}'.format(train_x.shape))
# print(train_x[:2, :])
# initial

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