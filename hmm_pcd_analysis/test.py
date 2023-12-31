import numpy as np
import data_loader
"""
隐马尔科夫模型实现，封装到ml_models.pgm
"""
start_pron = np.array([0.3, 0.6, 0.1]).T  # pi
trans_state = np.array([[0.8, 0.1, 0.1],
                    [0.1, 0.8, 0.1],
                    [0.1, 0.1, 0.8]])
emission = np.array([[0.76, 0.01, 0.01, 0.1, 0.1, 0.01, 0.01],
                      [0.01, 0.76, 0.01, 0.1, 0.1, 0.01, 0.01],
                      [0.01, 0.01, 0.76, 0.1, 0.1, 0.01, 0.01]])
states = ["none", "stone", "sand"]
vocabulary = [0,1,2,3,4,5,6]  # obs state


class HMM(object):
    def __init__(self, hidden_status_num=None, visible_status_num=None):
        """
        :param hidden_status_num: 隐状态数
        :param visible_status_num: 观测状态数
        """
        self.hidden_status_num = hidden_status_num
        self.visible_status_num = visible_status_num
        # 定义HMM的参数
        self.pi = None  # 初始隐状态概率分布 shape:[hidden_status_num,1]
        self.A = None  # 隐状态转移概率矩阵 shape:[hidden_status_num,hidden_status_num]
        self.B = None  # 观测状态概率矩阵 shape:[hidden_status_num,visible_status_num]

    def predict_joint_visible_prob(self, visible_list=None, forward_type="forward"):
        """
        前向/后向算法计算观测序列出现的概率值
        :param visible_list:
        :param forward_type:forward前向，backward后向
        :return:
        """
        if forward_type == "forward":
            # 计算初始值
            alpha = self.pi * self.B[:, [visible_list[0]]]
            # 递推
            for step in range(1, len(visible_list)):
                alpha = self.A.T.dot(alpha) * self.B[:, [visible_list[step]]]
            # 求和
            return np.sum(alpha)
        else:
            # 计算初始值
            beta = np.ones_like(self.pi)
            # 递推
            for step in range(len(visible_list) - 2, -1, -1):
                beta = self.A.dot(self.B[:, [visible_list[step + 1]]] * beta)
            # 最后一步
            return np.sum(self.pi * self.B[:, [visible_list[0]]] * beta)

    def fit_with_hidden_status(self, visible_list, hidden_list):
        """
        包含隐状态的参数估计
        :param visible_list: [[],[],...,[]]
        :param hidden_list: [[],[],...,[]]
        :return:
        """
        self.pi = np.zeros(shape=(self.hidden_status_num, 1))
        self.A = np.zeros(shape=(self.hidden_status_num, self.hidden_status_num))
        self.B = np.zeros(shape=(self.hidden_status_num, self.visible_status_num))
        for i in range(0, len(visible_list)):
            visible_status = visible_list[i]
            hidden_status = hidden_list[i]
            self.pi[hidden_status[0]] += 1
            for j in range(0, len(hidden_status) - 1):
                self.A[hidden_status[j], hidden_status[j + 1]] += 1
                self.B[hidden_status[j], visible_status[j]] += 1
            self.B[hidden_status[j + 1], visible_status[j + 1]] += 1
        # 归一化
        self.pi = self.pi / np.sum(self.pi)
        self.A = self.A / np.sum(self.A, axis=0)
        self.B = self.B / np.sum(self.B, axis=0)

    def fit_without_hidden_status(self, visible_list=None, tol=1e-5, n_iter=10):
        """
        不包含隐状态的参数估计:Baum-Welch算法
        :param visible_list: [...]
        :param tol:当pi,A,B的增益值变化小于tol时终止
        :param n_iter:迭代次数
        :return:
        """
        # 初始化参数
        if self.pi is None:
            self.pi = np.ones(shape=(self.hidden_status_num, 1)) + np.random.random(size=(self.hidden_status_num, 1))
            self.pi = self.pi / np.sum(self.pi)
        if self.A is None:
            self.A = np.ones(shape=(self.hidden_status_num, self.hidden_status_num)) + np.random.random(
                size=(self.hidden_status_num, self.hidden_status_num))
            self.A = self.A / np.sum(self.A, axis=0)
        if self.B is None:
            self.B = np.ones(shape=(self.hidden_status_num, self.visible_status_num)) + np.random.random(
                size=(self.hidden_status_num, self.visible_status_num))
            self.B = self.B / np.sum(self.B, axis=0)
        for _ in range(0, n_iter):
            # 计算前向概率
            alphas = []
            alpha = self.pi * self.B[:, [visible_list[0]]]
            alphas.append(alpha)
            for step in range(1, len(visible_list)):
                alpha = self.A.T.dot(alpha) * self.B[:, [visible_list[step]]]
                alphas.append(alpha)
            # 计算后向概率
            betas = []
            beta = np.ones_like(self.pi)
            betas.append(beta)
            for step in range(len(visible_list) - 2, -1, -1):
                beta = self.A.dot(self.B[:, [visible_list[step + 1]]] * beta)
                betas.append(beta)
            betas.reverse()
            # 计算gamma值
            gammas = []
            for i in range(0, len(alphas)):
                gammas.append((alphas[i] * betas[i])[:, 0])
            gammas = np.asarray(gammas)
            # 计算xi值
            xi = np.zeros_like(self.A)
            for i in range(0, self.hidden_status_num):
                for j in range(0, self.hidden_status_num):
                    xi_i_j = 0.0
                    for t in range(0, len(visible_list) - 1):
                        xi_i_j += alphas[t][i][0] * self.A[i, j] * self.B[j, visible_list[t + 1]] * \
                                  betas[t + 1][j][0]
                    xi[i, j] = xi_i_j
            loss = 0.0  # 统计累计误差
            # 更新参数
            # 初始概率
            for i in range(0, self.hidden_status_num):
                new_pi_i = gammas[0][i]
                loss += np.abs(new_pi_i - self.pi[i][0])
                self.pi[i] = new_pi_i
            # 隐状态转移概率
            for i in range(0, self.hidden_status_num):
                for j in range(0, self.hidden_status_num):
                    new_a_i_j = xi[i, j] / np.sum(gammas[:, i][:-1])
                    loss += np.abs(new_a_i_j - self.A[i, j])
                    self.A[i, j] = new_a_i_j
            # 观测概率矩阵
            for j in range(0, self.hidden_status_num):
                for k in range(0, self.visible_status_num):
                    new_b_j_k = np.sum(gammas[:, j] * (np.asarray(visible_list) == k)) / np.sum(gammas[:, j])
                    loss += np.abs(new_b_j_k - self.B[j, k])
                    self.B[j, k] = new_b_j_k
            # 归一化
            self.pi = self.pi / np.sum(self.pi)
            self.A = self.A / np.sum(self.A, axis=0)
            self.B = self.B / np.sum(self.B, axis=0)
            if loss < tol:
                break


if __name__ == "__main__":
    model = HMM(hidden_status_num=3, visible_status_num=30)
    data = data_loader.PointDataLoader("F:\earth_rosbag\data\\test3\\r3live_4pixel\sample_bag0")
    data.down_sample_loader()

    model.A = trans_state
    model.B = emission
    model.pi = start_pron

    points = data.down_points_dir
    model.fit_without_hidden_status()
