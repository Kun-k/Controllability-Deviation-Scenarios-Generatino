import sys

# sys.path.append('/home1/rk19/Scenario_Generation/RL_Scenario_Generation/byH2O')
sys.path.append('E:/scenario_generation/RL_Scenario_Generation/byH2O')

import pandas as pd
import numpy as np
import math
import os
import traci
from sumolib import checkBinary
import time
from utils.car_dis_comput import dist_between_cars


def dataset(record_file, track_file, output_file, r):
    # TODO: 车辆长宽，需随时更新数据
    length = 5
    width = 1.8

    # 最多采1000步、240米
    dataset = {'observations': [],
               'actions': [],
               'rewards': [],
               'next_observations': [],
               'terminals': []}

    # 文件读取
    record_data = pd.DataFrame(pd.read_csv(record_file))
    trans_data = pd.DataFrame(pd.read_csv(track_file))

    # 车辆数量、id、车流时间步数量
    carid = list(trans_data.groupby(['id']).groups.keys())
    num_agents = len(carid)
    num_adv_agents = num_agents - 1
    max_size = float('inf')
    for i in range(num_agents):
        max_size = min(max_size, len(trans_data[trans_data.id == carid[i]]))
    max_size -= 1
    ego_id_index = np.random.randint(0, num_agents)
    # 读取车辆行驶数据
    start_timestep = trans_data.frame.iloc[0]
    # 纵向位置偏执量
    # lowlane = float(record_data["lowerLaneMarkings"][0].split(";")[0])
    lowlane = 2 * float(record_data["lowerLaneMarkings"][0].split(";")[0]) - \
              float(record_data["lowerLaneMarkings"][0].split(";")[1])
    # 横向位置偏执
    x_min = np.min(trans_data['x'].tolist()) - 5

    # 计算observation, terminal
    terminal = False
    for t in range(max_size):
        # observation
        data_curr = trans_data[trans_data.frame == t + start_timestep]
        observation = []
        for i in range(num_agents):
            speed_x_curr = data_curr["xVelocity"].iloc[i]
            speed_y_curr = data_curr["yVelocity"].iloc[i]
            x_curr = data_curr["x"].iloc[i] - x_min
            y_curr = data_curr["y"].iloc[i] - lowlane
            speed_curr = math.sqrt(speed_x_curr ** 2 + speed_y_curr ** 2)
            yaw_curr = math.atan2(speed_y_curr, speed_x_curr)
            observation.extend([x_curr, y_curr, speed_curr, yaw_curr])
            if x_curr > 240 or t >= 1000:
                terminal = True
        for i in range(4):
            observation.insert(i, observation.pop(ego_id_index * 4 + i))
        dataset['observations'].append(observation)
        # terminal
        if t == max_size - 1 or terminal:
            dataset['terminals'].append(True)
            max_size = t + 1
            break
        else:
            dataset['terminals'].append(False)
    # 计算actions, next_observations, reward
    dt = 0.04
    up_cut = [0.6 * 9.8 * dt, math.pi / 3 * dt]
    low_cut = [-0.8 * 9.8 * dt, -math.pi / 3 * dt]
    for t in range(max_size):
        action = []
        next_observation = []
        for i in range(num_agents):
            speed_curr = dataset['observations'][t][4 * i + 2]
            yaw_curr = dataset['observations'][t][4 * i + 3]
            if t != max_size - 1:
                speed_next = dataset['observations'][t + 1][4 * i + 2]
                yaw_next = dataset['observations'][t + 1][4 * i + 3]
                next_observation.extend(dataset['observations'][t + 1][4 * i: 4 * (i + 1)])
            else:
                if i == 0:
                    index = ego_id_index
                elif i == ego_id_index:
                    index = 0
                else:
                    index = i
                data_next = trans_data[trans_data.frame == t + start_timestep + 1]
                speed_x_next = data_next["xVelocity"].iloc[index]
                speed_y_next = data_next["yVelocity"].iloc[index]
                x_next = data_next["x"].iloc[index] - x_min
                y_next = data_next["y"].iloc[index] - lowlane
                speed_next = math.sqrt(speed_x_next ** 2 + speed_y_next ** 2)
                yaw_next = math.atan2(speed_y_next, speed_x_next)
                next_observation.extend([x_next, y_next, speed_next, yaw_next])
            delta_speed = speed_next - speed_curr
            delta_yaw = yaw_next - yaw_curr
            # delta_speed = up_cut[0] if delta_speed > up_cut[0] \
            #     else low_cut[0] if delta_speed < low_cut[0] \
            #     else delta_speed
            # delta_yaw = up_cut[1] if delta_yaw > up_cut[1] \
            #     else low_cut[1] if delta_yaw < low_cut[1] \
            #     else delta_yaw
            action_speed = (delta_speed - low_cut[0]) * 2 / (up_cut[0] - low_cut[0]) - 1
            action_yaw = (delta_yaw - low_cut[1]) * 2 / (up_cut[1] - low_cut[1]) - 1
            action.extend([action_speed, action_yaw])
        action.pop(0)
        action.pop(0)
        dataset['actions'].append(action)
        dataset['next_observations'].append(next_observation)

        # 计算回报
        ego_state = next_observation[0:4]
        adv_state = next_observation[4:]
        # r1
        if r == "r1":
            reward = 0  # 没有碰撞就没有奖励或惩罚
            dataset['rewards'].append(reward)

        # r2
        elif r[0:2] == "r2":
            ego_col_cost_record, adv_col_cost_record, adv_road_cost_record = float('inf'), float('inf'), float('inf')
            # TODO: 设置阈值
            bv_bv_thresh = 1.5
            bv_road_thresh = float("inf")
            a, b, c = list(map(float, r[3:].split('-')))

            for i in range(num_adv_agents):
                car_ego = [ego_state[0], ego_state[1],
                           length, width, ego_state[3]]
                car_adv = [adv_state[i * 4 + 0], adv_state[i * 4 + 1],
                           length, width, adv_state[i * 4 + 3]]
                dis_ego_adv = dist_between_cars(car_ego, car_adv)
                # dis_ego_adv = math.sqrt((ego_state[0] - adv_state[i * 4 + 0]) ** 2 +
                #                         (ego_state[1] - adv_state[i * 4 + 1]) ** 2)
                if dis_ego_adv < ego_col_cost_record:
                    ego_col_cost_record = dis_ego_adv
            ego_col_cost = ego_col_cost_record

            for i in range(num_adv_agents):
                for j in range(i + 1, num_adv_agents):
                    car_adv_j = [adv_state[j * 4 + 0], adv_state[j * 4 + 1],
                                 length, width, adv_state[j * 4 + 3]]
                    car_adv_i = [adv_state[i * 4 + 0], adv_state[i * 4 + 1],
                                 length, width, adv_state[i * 4 + 3]]
                    dis_adv_adv = dist_between_cars(car_adv_i, car_adv_j)

                    # dis_adv_adv = np.sqrt((adv_state[i * 4 + 0] - adv_state[j * 4 + 0]) ** 2 +
                    #                       (adv_state[i * 4 + 1] - adv_state[j * 4 + 1]) ** 2)
                    if dis_adv_adv < adv_col_cost_record:
                        adv_col_cost_record = dis_adv_adv
            adv_col_cost = min(adv_col_cost_record, bv_bv_thresh)

            road_up, road_low = 12, 0
            car_width = 1.8
            for i in range(num_adv_agents):
                y = adv_state[i * 4 + 1]
                dis_adv_road = min(road_up - (y + car_width / 2), (y - car_width / 2) - road_low)
                if dis_adv_road < adv_road_cost_record:
                    adv_road_cost_record = dis_adv_road
            adv_road_cost = min(adv_road_cost_record, bv_road_thresh)
            reward = - a * ego_col_cost + b * adv_col_cost + c * adv_road_cost
            # reward = -ego_col_cost + adv_col_cost + adv_road_cost
            dataset['rewards'].append(reward)
            # TODO: BV距离AV太远则不再采样
            if ego_col_cost > 15:
                dataset['observations'] = dataset['observations'][0: t + 1]
                dataset['terminals'] = dataset['terminals'][0: t + 1]
                break
        # r3
        elif r == "r3":
            ego_col_cost_record = float('inf')
            for i in range(num_adv_agents):
                car_ego = [ego_state[0], ego_state[1],
                           length, width, ego_state[3]]
                car_adv = [adv_state[i * 4 + 0], adv_state[i * 4 + 1],
                           length, width, adv_state[i * 4 + 3]]
                dis_ego_adv = dist_between_cars(car_ego, car_adv)
                # dis_ego_adv = math.sqrt((ego_state[0] - adv_state[i * 4 + 0]) ** 2 +
                #                         (ego_state[1] - adv_state[i * 4 + 1]) ** 2)
                if dis_ego_adv < ego_col_cost_record:
                    ego_col_cost_record = dis_ego_adv
            ego_col_cost = ego_col_cost_record
            reward = -ego_col_cost
            dataset['rewards'].append(reward)
            if ego_col_cost > 15:
                dataset['observations'] = dataset['observations'][0: t + 1]
                dataset['terminals'] = dataset['terminals'][0: t + 1]
                break

    # print(dataset['observations'][0][0] - dataset['observations'][0][4])
    np.save(output_file, dataset)
    return dataset


def sim_by_dataset(filepath):
    states = np.load(filepath, allow_pickle='TRUE').item()['observations']
    rewards = np.load(filepath, allow_pickle='TRUE').item()['rewards']
    actions = np.load(filepath, allow_pickle='TRUE').item()['actions']

    t = 0
    cfg_sumo = '../config/lane.sumocfg'
    sim_seed = 42
    app = "sumo-gui"
    command = [checkBinary(app), '-c', cfg_sumo]
    command += ['--routing-algorithm', 'dijkstra']
    # command += ['--collision.action', 'remove']
    command += ['--seed', str(sim_seed)]
    command += ['--no-step-log', 'True']
    command += ['--time-to-teleport', '300']
    command += ['--no-warnings', 'True']
    command += ['--duration-log.disable', 'True']
    command += ['--waiting-time-memory', '1000']
    command += ['--eager-insert', 'True']
    command += ['--lanechange.duration', '2']
    command += ['--lateral-resolution', '0.0']
    traci.start(command)
    traci.simulationStep()
    for t in range(len(states)):
        state = states[t]
        for id in range(int(len(state) / 4)):
            [x, y, speed, yaw] = state[id * 4: (id + 1) * 4]
            car_type = "AV" if id == 0 else "BV"
            if t == 0:
                traci.vehicle.add(
                    vehID="car" + str(id),
                    routeID="straight",
                    typeID=car_type,
                    depart=10,
                    departLane="best",
                    departPos=5.0,
                    departSpeed=10,
                )
            traci.vehicle.moveToXY(vehID="car" + str(int(id)),
                                   x=x, y=y, angle=-yaw * 180 / np.pi + 90,
                                   edgeID=0, lane=0)
        traci.simulationStep()
        print(rewards[t], actions[t])
        time.sleep(0.04)
    traci.close()


if __name__ == "__main__":
    # sim_by_dataset("r2_dis_10_car_2/02/02_2_10_903.npy")
    r_list = [
        # "r1",
        "r2-1-1-1",
        "r2-0-1-1",
        "r2-1-2-1",
        "r2-1-1-0",
        "r2-0-1-0",
        "r2-1-2-0",
        # "r3"
    ]
    name_list = [
        # "dis_10_car_2",
        # "dis_20_car_3",
        # "dis_20_car_4",
        # "dis_25_car_5",
        "dis_25_car_6",
        # "dis_25_car_7",
    ]
    road_3_filenum = [1, 2, 3, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24]  # 3车道highD数据
    for r in r_list:
        for name in name_list:
            print(r + "_" + name)
            savepath_root = r + "_" + name
            if os.path.exists(savepath_root) is False:
                os.mkdir(savepath_root)
            for filenum in road_3_filenum:
                rootpath_highd = "E:/highD-dataset/data/"
                rootpath_cut = "E:/highD-dataset/highd_cut/"
                # rootpath_highd = "/home1/rk19/highD-dataset/data/"
                # rootpath_cut = "/home1/rk19/highD-dataset/highd_cut/"
                root_datapath = rootpath_cut + name + "/%02d_track_cut/" % filenum
                record_file = rootpath_highd + "%02d_recordingMeta.csv" % filenum
                # root_datapath = "/home1/rk19/highD-dataset/data/" + name  + "/%02d_track_cut/" % filenum
                # record_file = "/home1/rk19/highD-dataset/data/%02d_recordingMeta.csv" % filenum
                file_name_list = os.listdir(root_datapath)
                savepath = savepath_root + "/%02d/" % filenum
                if os.path.exists(savepath) is False:
                    os.mkdir(savepath)
                # os.mkdir("data/%02d" % filenum)
                for file_name in file_name_list:
                    track_file = root_datapath + file_name
                    output_file = savepath + file_name.split('.')[0]
                    dataset(record_file, track_file, output_file, r)
                    print(output_file)

    # r = "r2"
    # name = "dis_10_car_2"
    # road_3_filenum = [1, 2, 3, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24]  # 3车道highD数据
    # if os.path.exists(r + "_" + name) is False:
    #     os.mkdir(r + "_" + name)
    # for filenum in road_3_filenum:
    #     root_datapath = "E:/highD-dataset/data/" + name +"/%02d_track_cut/" % filenum
    #     record_file = "E:/highD-dataset/data/%02d_recordingMeta.csv" % filenum
    #     file_name_list = os.listdir(root_datapath)
    #     savepath = r + "_" + name + "/%02d/" % filenum
    #     if os.path.exists(savepath) is False:
    #         os.mkdir(savepath)
    #     # os.mkdir("data/%02d" % filenum)
    #     for file_name in file_name_list:
    #         track_file = root_datapath + file_name
    #         output_file = savepath + file_name.split('.')[0]
    #         dataset(record_file, track_file, output_file, r)
    #         print(output_file)
