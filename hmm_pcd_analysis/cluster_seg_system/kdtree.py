import os
import time
import numpy as np
import matplotlib.pyplot as plt


class Node:
    def __init__(self, point, axis):
        self.point = point  # [x, y, z, index]
        self.left = None
        self.right = None
        self.axis = axis

        self.D2F = None  # the distance to the father point
        self.grad2F = None


class KDTree:
    def __init__(self, points_input):
        """2d-tree"""
        self.points_input = points_input
        self.tree = self.build_kdtree(points_input)

    def build_kdtree(self, points, father_point=None, depth=0, k=2):
        """the input points: [[x, y, z, index], ]"""
        if len(points) == 0:
            return None

        axis = depth % k
        sorted_points = sorted(points, key=lambda point: point[axis])
        median_idx = len(points) // 2

        node = Node(sorted_points[median_idx], axis)
        node.left = self.build_kdtree(sorted_points[:median_idx], sorted_points[median_idx], depth + 1)
        node.right = self.build_kdtree(sorted_points[median_idx + 1:], sorted_points[median_idx], depth + 1)
        if depth != 0 and father_point:
            node.D2F = distance(father_point[0:2], sorted_points[median_idx][0:2])
            node.grad2F = abs(father_point[2]-sorted_points[median_idx][2])/node.D2F

        return node

    def crack_detect(self, thread_hold=20):
        """ find the crack grad in the tree """
        for element in self.in_order_traversal(self.tree):
            if element and element >= thread_hold:
                print(element)
                return True

    def in_order_traversal(self, node):
        """traverse the elements in the kdtree"""
        if node is not None:
            yield from self.in_order_traversal(node.left)
            yield node.grad2F  # output the grad
            yield from self.in_order_traversal(node.right)

    def kdtree_visualize(self):
        fig, ax = plt.subplots()
        self._plot_kd_tree(ax, self.tree, 0, 10, 0, 8)
        self._plot_points(ax, self.points_input)

        plt.xlabel('X-axis')
        plt.ylabel('Y-axis')
        plt.title('KD-tree Visualization')
        plt.show()

    def _plot_kd_tree(self, ax, node, xmin, xmax, ymin, ymax, depth=0):
        """Recursive drawing"""
        if node is not None:
            if node.axis == 0:
                ax.plot([node.point[0], node.point[0]], [ymin, ymax], color='black')
                self._plot_kd_tree(ax, node.left, xmin, node.point[0], ymin, ymax, depth + 1)
                self._plot_kd_tree(ax, node.right, node.point[0], xmax, ymin, ymax, depth + 1)
            else:
                ax.plot([xmin, xmax], [node.point[1], node.point[1]], color='black')
                self._plot_kd_tree(ax, node.left, xmin, xmax, ymin, node.point[1], depth + 1)
                self._plot_kd_tree(ax, node.right, xmin, xmax, node.point[1], ymax, depth + 1)

    def _plot_points(self, ax, points):
        x_values = []
        y_values = []
        for p in points:
            x_values.append(p[0])
            y_values.append(p[1])
        ax.scatter(x_values, y_values, color='red', marker='o')
        for i, point in enumerate(points):
            ax.text(point[0], point[1], point[3], fontsize=10, ha='center', va='center')


def distance(point1, point2):
    return np.sqrt(np.sum((np.array(point1) - np.array(point2)) ** 2))


def closest_point_brute_force(points, target):
    return min(points, key=lambda x: distance(x, target))


def closest_point_kd_tree(root, target, depth=0, best=None):
    if root is None:
        return best

    k = len(target)
    axis = depth % k

    next_best = None
    next_branch = None

    if best is None or distance(root.point, target) < distance(best.point, target):
        next_best = root
    else:
        next_best = best

    if target[axis] < root.point[axis]:
        next_branch = root.left
    else:
        next_branch = root.right

    return closest_point_kd_tree(next_branch, target, depth + 1, next_best)


def grad(point1, point2):
    """point1: [x, y, z] array"""
    g = (point2[3]-point1[3])/np.linalg.norm( point2[0:2]-point1[0:2] )  # add abs
    return g


if __name__ == "__main__":
    # Example usage
    points = [[2,3, 10, 52], [5,4, 14, 53], [9,6, 9, 54], [4,7, 8, 55], [8,1, 19, 56], [7,2, 24, 57]]
    start_time = time.time()
    tree = KDTree(points)
    print("the 2d-tree has been built")
    tree.crack_detect()
    end_time = time.time()
    print(f"Execution Time: {(end_time - start_time):.6f} seconds")