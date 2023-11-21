'''
  Replay buffer for storing samples.
  @python version : 3.6.8
'''
import torch
import numpy as np
import os


class Buffer(object):

    def append(self, *args):
        pass

    def sample(self, *args):
        pass

class ReplayBuffer(Buffer):
    def __init__(self, state_dim, action_dim, max_size=int(1e6), device='cuda', datapath=None):
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

