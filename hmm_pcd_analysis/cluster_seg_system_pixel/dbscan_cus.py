import numpy as np
import time
import matplotlib.pyplot as plt


class DBSCAN_CUS:
    def __init__(self, eps, min_samples, x):
        self.eps = eps
        self.min_samples = min_samples

        self.n_samples = len(x)
        self.label_list = np.zeros(self.n_samples)  # list
        self._sort_data(np.array(x))

    def fit(self):
        """
        X 要进过排序的，或者将排序工作也置入dbscan的其中一个函数中
        分类 id 从 0 没处理，-1 为噪声，1开始为类
        返回原点list顺序的类型list [0 0 -1 1 1 1 ...]
        """
        # init
        clusterId = 0
        back = 0
        base = 0
        front = 0
        # while front < self.n_samples-1:
        while base < self.n_samples-1:
            if self.sort_list[front+1] - self.sort_list[base] <= self.eps:  # 未来推进状态
                if front < self.n_samples-2:
                    front += 1
                    continue
                else:  # 为了收尾
                    back_n, front_n, all_n = self._point_count(back, base, front)
                    if all_n >= self.min_samples:
                        self._update_label(back, front+1, clusterId)
                    else:
                        self._update_label(back, front + 1, -1)
                    base = front + 1
            else:
                back_n, front_n, all_n = self._point_count(back, base, front)
                if all_n >= self.min_samples:  # key point
                    # ket point 周围能连接的都是同簇
                    self._update_label(back, front, clusterId)
                    if front_n >= self.min_samples:
                        # base, back index jump update
                        back = self._back_update(base, front)
                        base = front
                    else:
                        # base, back index update
                        base += 1
                        back = self._back_update(back, base)
                        if base == front+1:
                            clusterId += 1
                else:
                    self._update_label(back, front, -1)
                    base += 1
                    back = self._back_update(back, base)

        return np.int64(self.label_list)

    def adaptive_fit(self):
        return None

    def _update_label(self, back, front, clusterId):
        # 连续点label, 两端点都都幅值
        for index in self.sort_index[back:front + 1]:
            self.label_list[index] = clusterId

    def _point_count(self, back, base, front):
        back_count = base - back + 1
        front_count = front - base + 1
        all_count = front - back + 1
        return back_count, front_count, all_count

    def _back_update(self, start, end):
        for i in range(start, end + 1):
            if self.sort_list[end] - self.sort_list[i] <= self.eps:
                return int(i)  # 可以等于base
            else:
                continue

    def _sort_data(self, x):
        self.depth_list = x  # original data np
        self.sort_list = np.sort(self.depth_list)  # 从小到大的数字
        # sort_index 序号的list [5,2,3,0,4,1], 排最小的数在原来的序号5的点
        self.sort_index = np.argsort(self.depth_list)


if __name__ == "__main__":
    path = 'F:\cluster_seg_system_pixel\output.txt'
    cluster_data = np.loadtxt(path, delimiter=',')
    cluster_data[120] = 8.3
    cluster_data[121] = 8.33
    s = time.time()
    cluster = DBSCAN_CUS(0.2, 5, cluster_data)
    res = cluster.fit()
    print(time.time() - s)

    x = res.tolist()
    plt.scatter(res.tolist(), cluster_data, marker='x', label='Crosses')
    plt.xlabel('cluster class-axis')
    plt.ylabel('depth-axis')
    plt.title('Scatter Plot of Points')
    plt.legend()
    plt.xticks(range(int(min(x)), int(max(x)) + 1, 1))
    plt.grid(True)
    plt.show()
