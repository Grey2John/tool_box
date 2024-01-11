import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import Delaunay
from itertools import combinations
import time


def delaunay_crack_detect(point_list, thread_hold=10):
    """go through all the edge, find the grad > thread_hold
    input: point_list [xy, z, index]"""
    if len(point_list) < 3:
        return False

    points = np.array([[i[0], i[1]] for i in point_list])
    triangulation = Delaunay(points)
    # delaunay_visual(triangulation, points)

    sorted_simplices = np.sort(triangulation.simplices, axis=1)
    all_combinations = []
    for matrix in sorted_simplices:
        matrix_combinations = list(combinations(matrix, 2))
        all_combinations.extend(matrix_combinations)
    # grid_list = []
    for j in all_combinations:
        g = grad_cal(point_list, j)
        if g >= thread_hold:
            # print(element)
            return True
    return False


def delaunay_visual(triangulation, points):
    plt.plot(points[:, 0], points[:, 1], 'o', label='Original Points')

    # Plot the Delaunay triangulation
    plt.triplot(points[:, 0], points[:, 1], triangulation.simplices, linewidth=0.5, color='red')

    # Label the triangles
    for j, p in enumerate(points):
        plt.text(p[0], p[1], f'{j}', ha='right', va='bottom')

    plt.xlabel('X-axis')
    plt.ylabel('Y-axis')
    plt.title('Delaunay Triangulation')
    plt.legend()
    plt.show()


def grad_cal(point_list, two_p_index):
    D2F = distance(point_list[two_p_index[0]][0:2], point_list[two_p_index[1]][0:2])
    grad = abs(point_list[two_p_index[0]][2] - point_list[two_p_index[1]][2])/D2F
    return grad


def distance(point1, point2):
    return np.sqrt(np.sum((np.array(point1) - np.array(point2)) ** 2))


# triangulation.simplices 是3*三角形数，3个点的索引
if __name__ == "__main__":
    point_list = [[538.5732722363833, 424.56295347067874, 73.82329471132304, 980],
                   [538.7979569781179, 422.6144060015364, 74.41742518445736, 4032],
                   [538.4909684551654, 424.857862984168, 76.40096325899789, 4925],
                   [544.860217320319, 424.55840955624575, 74.7152817408008, 49],
                   [547.4757348148888, 425.8483104491301, 76.39875904758786, 1994],
                   [544.1926193055886, 423.37345757999685, 74.81055166978896, 2440],
                   [542.5559431778186, 423.4790398417264, 76.17090343774034, 3618],
                   [549.5293351977791, 426.2430489520506, 76.24794424723883, 2762],
                   [540.5377484994428, 427.94195412829265, 74.67712088187935, 3025],
                   [542.191885408697, 428.20873765726435, 74.84657180658435, 3339],
                   [537.6097583427877, 430.49287427513997, 73.48085165058956, 4031],
                   [538.1929585870046, 428.12786198511714, 76.18035693529828, 4226],
                   [542.83084141995, 431.00121148318163, 73.65792394921984, 979],
                   [544.576509977827, 431.33525577681206, 74.24841338956989, 3617],
                   [550.5096752049791, 429.62242269804716, 75.2173187325399, 50],
                   [547.7807430965308, 430.2414584078233, 74.27088127684463, 2439],
                   [550.0377401486345, 429.5812554676778, 75.91260305312622, 3891],
                   [538.9049729013728, 436.3433134374719, 74.0256918069853, 4225],
                   [543.3765624349732, 435.4030680095237, 73.37608276305534, 3024],
                   [545.4507159845526, 435.46152678052226, 73.5495955786509, 3340]]
    points = np.array([[i[0], i[1]] for i in point_list])
    triangulation = Delaunay(points)
    print("done")