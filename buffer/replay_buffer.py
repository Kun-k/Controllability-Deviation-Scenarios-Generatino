'''
  Replay buffer for storing samples.
  @python version : 3.6.8
'''
import torch
import numpy as np
import os
from utils.car_dis_comput import dist_between_cars
from utils.collision_test import col_test


class Buffer(object):

    def append(self, *args):
        pass

    def sample(self, *args):
        pass

class ReplayBuffer(Buffer):
    def __init__(self, state_dim, action_dim, max_size=int(1e6), device='cuda', datapath=None, reward_fun=None,
                 car_info=None, road_info=None):
        if datapath != None:
            file_list = []
            for f in os.listdir(datapath):
                for ff in os.listdir(datapath + f):
                    file_list.append(datapath + f + '/' + ff)

            dataset = {'observations': [],
                       'actions': [],
                       'rewards': [],
                       'next_observations': [],
                       'terminals': []}
            key_list = ['observations', 'actions', 'rewards', 'next_observations', 'terminals']

            for r in range(len(file_list)):
                newdataset = np.load(file_list[r], allow_pickle=True).item()
                for key in key_list:
                    dataset[key].extend(newdataset[key])

            self.dataset = dataset
            self.action_dim = action_dim
            self.reward_fun = self.adv_r1 if reward_fun == "adv_r1" \
                else self.adv_r2 if reward_fun[0: 6] == "adv_r2" \
                else self.adv_r4 if reward_fun[0: 6] == "adv_r4" \
                else self.ego_r1 if reward_fun == "ego_r1" \
                else None

            if self.reward_fun is not None:
                self.car_info = car_info
                self.road_info = road_info
                self.max_speed = 40
                # TODO: 应该只对特定数据计算
                self.reward_fun()

            total_num = len(dataset['observations'])
            idx = np.random.choice(range(total_num), total_num, replace=False)
            self.state = np.vstack(np.array(dataset['observations'])).astype(np.float32)[idx, :
                ]  # An (N, dim_observation)-dimensional numpy array of observations
            self.action = np.vstack(np.array(dataset['actions'])).astype(np.float32)[idx,
                :]  # An (N, dim_action)-dimensional numpy array of actions
            self.reward = np.vstack(np.array(dataset['rewards'])).astype(np.float32)[idx,
                :]  # An (N,)-dimensional numpy array of rewards
            self.next_state = np.vstack(np.array(dataset['next_observations'])).astype(np.float32)[idx,
                 :]  # An (N, dim_observation)-dimensional numpy array of next observations
            self.done = np.vstack(np.array(dataset['terminals']))[idx,
                   :]  # An (N,)-dimensional numpy array of terminal flags
            fixed_dataset_size = self.reward.shape[0]
            self.max_size = fixed_dataset_size
            self.ptr = fixed_dataset_size
            self.size = fixed_dataset_size
        else:
            self.state = np.zeros((max_size, state_dim))
            self.action = np.zeros((max_size, action_dim))
            self.next_state = np.zeros((max_size, state_dim))
            self.reward = np.zeros((max_size, 1))
            self.done = np.zeros((max_size, 1))
            self.max_size = max_size
            self.ptr = 0
            self.size = 0

        self.device = torch.device(device)

    def append(self, state, action, reward, next_state, done):
        self.state[self.ptr] = state
        self.action[self.ptr] = action
        self.next_state[self.ptr] = next_state
        self.reward[self.ptr] = reward
        self.done[self.ptr] = done

        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    def add_batch(self,
            obss: np.ndarray,
            next_obss: np.ndarray,
            actions: np.ndarray,
            rewards: np.ndarray,
            terminals: np.ndarray):

        batch_size = len(obss)
        indexes = np.arange(self.ptr, self.ptr + batch_size) % self.max_size

        self.state[indexes] = np.array(obss).copy()
        self.next_state[indexes] = np.array(next_obss).copy()
        self.action[indexes] = np.array(actions).copy()
        self.reward[indexes] = np.array(rewards).copy()
        self.done[indexes] = np.array(terminals).copy().reshape(-1, 1)

        self.ptr = (self.ptr + batch_size) % self.max_size
        self.size = min(self.size + batch_size, self.max_size)

    def sample(self, batch_size):
        ind = np.random.randint(0, self.size, size=batch_size)

        return {
            'observations': torch.FloatTensor(self.state[ind]).to(self.device), 
            'actions': torch.FloatTensor(self.action[ind]).to(self.device), 
            'rewards': torch.FloatTensor(self.reward[ind]).to(self.device), 
            'next_observations': torch.FloatTensor(self.next_state[ind]).to(self.device), 
            'dones': torch.FloatTensor(self.done[ind]).to(self.device)
            }

    def sample_all(self):
        return {
            'observations': torch.FloatTensor(self.state).to(self.device),
            'actions': torch.FloatTensor(self.action).to(self.device),
            'rewards': torch.FloatTensor(self.reward).to(self.device),
            'next_observations': torch.FloatTensor(self.next_state).to(self.device),
            'dones': torch.FloatTensor(self.done).to(self.device)
        }

    #### TODO: 设置几组参数可调的奖励函数
    def adv_r2(self):
        self.dataset['rewards'] = []
        # length = 5
        # width = 1.8
        [length_ego, width_ego] = self.car_info[0][2]
        num_adv_agents = int(self.action_dim / 2)
        for t in range(len(self.dataset['observations'])):
            ego_state = self.dataset['next_observations'][t][0:4]
            adv_state = self.dataset['next_observations'][t][4:]
            # r2
            ego_col_cost_record, adv_col_cost_record, adv_road_cost_record = float('inf'), float(
                'inf'), float(
                'inf')
            bv_bv_thresh = 1.5
            bv_road_thresh = float("inf")
            a, b, c = list(map(float, self.reward_fun[3:].split('-')))

            for i in range(num_adv_agents):
                [length, width] = self.car_info[i + 1][2]
                car_ego = [ego_state[0], ego_state[1],
                           length_ego, width_ego, ego_state[3]]
                careward_fun = [adv_state[i * 4 + 0], adv_state[i * 4 + 1],
                                length, width, adv_state[i * 4 + 3]]
                dis_ego_adv = dist_between_cars(car_ego, careward_fun)
                if dis_ego_adv < ego_col_cost_record:
                    ego_col_cost_record = dis_ego_adv
            ego_col_cost = ego_col_cost_record

            for i in range(num_adv_agents):
                for j in range(i + 1, num_adv_agents):
                    careward_fun_j = [adv_state[j * 4 + 0], adv_state[j * 4 + 1],
                                      length, width, adv_state[j * 4 + 3]]
                    careward_fun_i = [adv_state[i * 4 + 0], adv_state[i * 4 + 1],
                                      length, width, adv_state[i * 4 + 3]]
                    dis_adv_adv = dist_between_cars(careward_fun_i, careward_fun_j)
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
            col = col_test(self.car_info, self.dataset['next_observations'][t], self.road_info)
            col_list = col[0] + col[1]
            reward += 100 if 0 in col_list \
                else -100 if len(col_list) != 0 \
                else 0
            self.dataset['rewards'].append(reward)
            # if ego_col_cost > 15:
            #     self.dataset['observations'] = self.dataset['observations'][0: t + 1]
            #     self.dataset['terminals'] = self.dataset['terminals'][0: t + 1]
            #     break

    def adv_r1(self):
        self.dataset['rewards'] = []
        for t in range(len(self.dataset['observations'])):
            col = col_test(self.car_info, self.dataset['next_observations'][t], self.road_info)
            col_list = col[0] + col[1]
            reward = 100 if 0 in col_list \
                else -100 if len(col_list) != 0 \
                else 0
            self.dataset['rewards'].append(reward)

    # todo: 为adv_r4和ego_r1设置可调的参数
    def adv_r4(self):
        self.dataset['rewards'] = []
        # TODO: self.max_speed
        num_adv_agents = int(self.action_dim / 2)
        for t in range(len(self.dataset['observations'])):
            adv_state = self.dataset['next_observations'][t][4:]
            col = col_test(self.car_info, self.dataset['next_observations'][t], self.road_info)
            col_list = col[0] + col[1]
            col_cost = 100 if 0 in col_list \
                else -100 if len(col_list) != 0 \
                else 0
            speed_cost = 0
            yaw_cost = 0
            for i in range(num_adv_agents):
                speed_cost = adv_state[i: (i + 1) * 4][2] / self.max_speed - 1 / 2
                yaw_cost = - abs(adv_state[i: (i + 1) * 4][3]) / (np.pi / 3) * 5 * 0
            reward = col_cost + speed_cost + yaw_cost
            self.dataset['rewards'].append(reward)

    def ego_r1(self):
        self.dataset['rewards'] = []
        # TODO: self.max_speed
        for t in range(len(self.dataset['observations'])):
            ego_state = self.dataset['next_observations'][t][0:4]
            # TODO: 碰撞奖励
            # col_cost_ego = -20 if 0 in self.CollidingVehs else 0
            col = col_test(self.car_info, self.dataset['next_observations'][t], self.road_info)
            col_list = col[0] + col[1]
            col_cost_ego = -20 if 0 in col_list else 0
            speed_cost_ego = ego_state[2] / self.max_speed - 1 / 2
            yaw_cost_ego = - abs(ego_state[3]) / (np.pi / 3) * 5 * 0
            reward = col_cost_ego + speed_cost_ego + yaw_cost_ego
            self.dataset['rewards'].append(reward)


def batch_to_torch(batch, device):
    return {
        k: torch.from_numpy(v).to(device=device, non_blocking=True)
        for k, v in batch.items()
    }


# def get_d4rl_dataset(env):
#     dataset = d4rl.qlearning_dataset(env)
#     return dict(
#         observations=dataset['observations'],
#         actions=dataset['actions'],
#         next_observations=dataset['next_observations'],
#         rewards=dataset['rewards'],
#         dones=dataset['terminals'].astype(np.float32),
#     )


def index_batch(batch, indices):
    indexed = {}
    for key in batch.keys():
        indexed[key] = batch[key][indices, ...]
    return indexed


def parition_batch_train_test(batch, train_ratio):
    train_indices = np.random.rand(batch['observations'].shape[0]) < train_ratio
    train_batch = index_batch(batch, train_indices)
    test_batch = index_batch(batch, ~train_indices)
    return train_batch, test_batch


def subsample_batch(batch, size):
    indices = np.random.randint(batch['observations'].shape[0], size=size)
    return index_batch(batch, indices)


def concatenate_batches(batches):
    concatenated = {}
    for key in batches[0].keys():
        concatenated[key] = np.concatenate([batch[key] for batch in batches], axis=0).astype(np.float32)
    return concatenated


def split_batch(batch, batch_size):
    batches = []
    length = batch['observations'].shape[0]
    keys = batch.keys()
    for start in range(0, length, batch_size):
        end = min(start + batch_size, length)
        batches.append({key: batch[key][start:end, ...] for key in keys})
    return batches


def split_data_by_traj(data, max_traj_length):
    dones = data['dones'].astype(bool)
    start = 0
    splits = []
    for i, done in enumerate(dones):
        if i - start + 1 >= max_traj_length or done:
            splits.append(index_batch(data, slice(start, i + 1)))
            start = i + 1

    if start < len(dones):
        splits.append(index_batch(data, slice(start, None)))

    return splits

