import math
import numpy as np
from copy import deepcopy


def col_test(car_info, car_state, road_info):
    """
    Info:
        直线道路上的车辆碰撞检测、车辆驶出路面检测;
        在原版本上, 将car_state更改为H2O中的形式.

    Args:
        car_info (list): 所有车辆的信息, [[ID(int), lane(str), [length(float), width(float)]], ...]
        car_state (list): 所有车辆的位置, [[x(float), y(float), speed(float), yaw(float)], ...]
        road_info (dict): 道路信息, {lane(str): [minLaneMarking(float), maxLaneMarking(float)], ...}

    Return:
        ...
    """
    matrix_car_list = []  # 记录车辆四个顶点的位置
    col_car_car_list = []  # 记录车车碰撞信息
    col_car_road_list = []  # 记录车路碰撞信息
    # 车辆驶出路面检测
    # 可判断车辆是否压两侧车道线, 但如何应对车辆完全在道路外的情况
    for i in range(len(car_info)):
        [ID, lane, [length, width]] = car_info[i]
        [x, y, speed, yaw] = car_state[i * 4: (i + 1) * 4]
        [minLaneMarking, maxLaneMarking] = road_info[lane]
        sin_yaw = math.sin(yaw)
        cos_yaw = math.cos(yaw)
        matrix_car = [[x - width / 2 * sin_yaw - length * cos_yaw, y + width / 2 * cos_yaw - length * sin_yaw],
                      [x - width / 2 * sin_yaw, y + width / 2 * cos_yaw],
                      [x + width / 2 * sin_yaw, y - width / 2 * cos_yaw],
                      [x + width / 2 * sin_yaw - length * cos_yaw, y - width / 2 * cos_yaw - length * sin_yaw]]
        matrix_car = np.array(matrix_car)
        matrix_car_list.append(matrix_car)
        y_max = np.max(matrix_car[:, 1])
        y_min = np.min(matrix_car[:, 1])
        if y_min < minLaneMarking < y_max or y_min < maxLaneMarking < y_max:
            col_car_road_list.append(ID)
    # 车辆碰撞检测
    # 通过叉乘判断是否线段相交，进而判断是否碰撞
    # 参考: https://blog.csdn.net/m0_37660632/article/details/123925503
    # 求向量ab和向量cd的叉乘
    def xmult(a, b, c, d):
        vectorAx = b[0] - a[0]
        vectorAy = b[1] - a[1]
        vectorBx = d[0] - c[0]
        vectorBy = d[1] - c[1]
        return (vectorAx * vectorBy - vectorAy * vectorBx)

    car_info_ = deepcopy(car_info)
    while len(car_info_) != 0:
        ID_i = car_info_.pop(0)[0]
        matrix_car_i = matrix_car_list.pop(0)
        j = 0
        while j < len(car_info_):
            ID_j = car_info_[j][0]
            matrix_car_j = matrix_car_list[j]
            collision = False
            for p in range(-1, 3):
                c, d = matrix_car_i[p], matrix_car_i[p + 1]
                for q in range(-1, 3):
                    a, b = matrix_car_j[q], matrix_car_j[q + 1]
                    xmult1 = xmult(c, d, c, a)
                    xmult2 = xmult(c, d, c, b)
                    xmult3 = xmult(a, b, a, c)
                    xmult4 = xmult(a, b, a, d)
                    if xmult1 * xmult2 < 0 and xmult3 * xmult4 < 0:
                        collision = True
                        break
                if collision:
                    break
            if collision:
                if ID_i not in col_car_car_list:
                    col_car_car_list.append(ID_i)
                if ID_j not in col_car_car_list:
                    col_car_car_list.append(ID_j)
            j += 1
    return col_car_car_list, col_car_road_list


if __name__ == "__main__":
    car_info = [[0, "lane0", [4, 2]],
                [1, "lane1", [4, 2]],
                [2, "lane0", [4, 2]],
                [3, "lane1", [4, 2]],
                [4, "lane1", [4, 2]]]
    car_state = [20, 1.5, 10, 5 * math.pi / 180,
                 30, 9.5, 10, 0 * math.pi / 180,
                 35, 0.8, 10, -30 * math.pi / 180,
                 10, 8, 10, 0 * math.pi / 180,
                 5.9, 8.5, 10, 7 * math.pi / 180]
    road_info = {"lane0": [0, 4], "lane1": [6, 10]}
    col_car_car_list, col_car_road_list = col_test(car_info, car_state, road_info)
    print(col_car_car_list)
    print(col_car_road_list)
