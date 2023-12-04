import datetime
import os
import sys
from copy import deepcopy
import absl.app
import absl.flags
import numpy as np
import wandb
from tqdm import trange
import torch
from models.sac import SAC
from envs.envs import Env
from buffer.replay_buffer import ReplayBuffer, batch_to_torch, subsample_batch
from models.model import TanhGaussianPolicy, FullyConnectedQFunction, SamplerPolicy
from buffer.sampler import StepSampler, TrajSampler
from utils.utils import Timer, define_flags_with_default, set_random_seed, get_user_flags, prefix_metrics
from utils.utils import WandBLogger
from utils.viskit.logging import logger, setup_logger
from models.conservative_sac import ConservativeSAC

import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--USED_wandb', type=str, default="False")
parser.add_argument('--ego_policy', type=str, default="sumo")  # "uniform", "sumo", "fvdm"
parser.add_argument('--adv_policy', type=str, default="RL")  # "uniform", "sumo", "fvdm"
parser.add_argument('--num_agents', type=int, default=1)
parser.add_argument('--r_ego', type=str, default="r1")
parser.add_argument('--r_adv', type=str, default="r1")
parser.add_argument('--realdata_path', type=str, default="E:/scenario_generation/dataset/Re_2_H2O/r3_dis_10_car_2/")  #
# parser.add_argument('--realdata_path', type=str, default="../byH2O/dataset/r1_dis_10_car_2/")  #
# parser.add_argument('--batch_ratio', type=float, default=0.5)
parser.add_argument('--is_save', type=str, default="False")
parser.add_argument('--device', type=str, default="cuda:0")
parser.add_argument('--seed', type=int, default=42)
parser.add_argument('--save_model', type=str, default="False")
# parser.add_argument('--cql_min_q_weight', type=float, default=0.1)
parser.add_argument('--alpha_l2', type=float, default=0.0)

args = parser.parse_args()
args.is_save = True if args.is_save == "True" else False
args.USED_wandb = True if args.USED_wandb == "True" else False
args.save_model = True if args.save_model == "True" else False
print(args)
while len(sys.argv) > 1:
	sys.argv.pop()
FLAGS_DEF = define_flags_with_default(
    alpha_l2=args.alpha_l2,

    USED_wandb = args.USED_wandb,
    ego_policy=args.ego_policy,  # "uniform", "sumo", "fvdm"
    adv_policy=args.adv_policy,  # "uniform", "sumo", "fvdm"
    num_agents=args.num_agents,
    r_ego=args.r_ego,
    r_adv=args.r_adv,
    realdata_path=args.realdata_path,
    # batch_ratio=args.batch_ratio,  # 仿真数据的比例
    is_save=args.is_save,
    device=args.device,
    # cql_min_q_weight=args.cql_min_q_weight,
    seed=args.seed,
    replay_buffer_size=1000000,

    current_time=datetime.datetime.now().strftime('%y-%m-%d-%H-%M-%S'),
    # variety_list="2.0",
    replaybuffer_ratio=10,
    real_residual_ratio=1.0,
    dis_dropout=False,
    max_traj_length=100,
    save_model=False,
    batch_size=256,

    reward_scale=1.0,
    reward_bias=0.0,
    clip_action=1.0,
    joint_noise_std=0.0,

    policy_arch='256-256',
    qf_arch='256-256',
    orthogonal_init=False,
    policy_log_std_multiplier=1.0,
    policy_log_std_offset=-1.0,

    # train and evaluate policy
    n_epochs_ego=0,
    n_epochs_adv=1000,
    n_loops=1,
    bc_epochs=0,
    n_rollout_steps_per_epoch=1000,
    n_train_step_per_epoch=1000,
    eval_period=10,
    eval_n_trajs=20,

    sac_ego=SAC.get_default_config(),
    cql_adv=ConservativeSAC.get_default_config(),
    logging=WandBLogger.get_default_config(),
)

def argparse():
    ...


# TODO:
# 打通数据流
# 增加新的loss

def main(argv):
    FLAGS = absl.flags.FLAGS
    # label = 'without_l2'
    label = 'CQL'
    run_name = f"{label}CQL_av={FLAGS.ego_policy}_" \
               f"bv={FLAGS.num_agents}-{FLAGS.adv_policy}_" \
               f"alpha_l2={FLAGS.alpha_l2}_" \
               f"r-ego={FLAGS.r_ego}_r-adv={FLAGS.r_adv}_" \
               f"seed={FLAGS.seed}_time={FLAGS.current_time}"
    if FLAGS.is_save:
        eval_savepath = "output/" + run_name + "/"
        if os.path.exists("output") is False:
            os.mkdir("output")
        if os.path.exists(eval_savepath) is False:
            os.mkdir(eval_savepath)
            os.mkdir(eval_savepath + "avcrash")
            os.mkdir(eval_savepath + "bvcrash")
            os.mkdir(eval_savepath + "avarrive")
    else:
        eval_savepath = None
    if FLAGS.USED_wandb:
        variant = get_user_flags(FLAGS, FLAGS_DEF)
        wandb_logger = WandBLogger(config=FLAGS.logging, variant=variant)
        wandb.run.name = run_name

        setup_logger(
            variant=variant,
            exp_id=wandb_logger.experiment_id,
            seed=FLAGS.seed,
            base_log_dir=FLAGS.logging.output_dir,
            include_exp_prefix_sub_dir=False
        )

    set_random_seed(FLAGS.seed)

    # real_env = Env(realdata_path=FLAGS.realdata_path, num_agents=FLAGS.num_agents, sim_horizon=FLAGS.max_traj_length,
    #                ego_policy=FLAGS.ego_policy, adv_policy=FLAGS.adv_policy,
    #                r_ego=FLAGS.r_ego, r_adv=FLAGS.r_adv, sim_seed=FLAGS.seed)
    # TODO: 关于奖励函数，这里使用adv_r4，要么在env里面改了，要么在采样后对数据全部重新处理
    sim_env = Env(realdata_path=FLAGS.realdata_path, num_agents=FLAGS.num_agents, sim_horizon=FLAGS.max_traj_length,
                   ego_policy=FLAGS.ego_policy, adv_policy=FLAGS.adv_policy,
                   r_ego=FLAGS.r_ego, r_adv=FLAGS.r_adv, sim_seed=FLAGS.seed)

    # TODO: real_env 和 sim_env 本质上没区别，考虑后续全部使用同一个
    train_sampler = StepSampler(sim_env, max_traj_length=FLAGS.max_traj_length)
    eval_sampler = TrajSampler(sim_env, rootsavepath=eval_savepath, max_traj_length=FLAGS.max_traj_length)
    num_state = sim_env.state_space[0]
    num_action_adv = sim_env.action_space_adv[0]

    replay_buffer_real = ReplayBuffer(num_state, num_action_adv, FLAGS.replay_buffer_size, device=FLAGS.device,
                                      datapath=FLAGS.realdata_path,
                                      reward_fun="adv_r4", car_info=sim_env.car_info, road_info=sim_env.road_info)
    replay_buffer_tempt = ReplayBuffer(num_state, num_action_adv, FLAGS.replay_buffer_size, device=FLAGS.device,
                                       reward_fun="adv_r4", car_info=sim_env.car_info, road_info=sim_env.road_info)
    replay_buffer_expend = ReplayBuffer(num_state, num_action_adv, FLAGS.replay_buffer_size, device=FLAGS.device,
                                        reward_fun="adv_r4", car_info=sim_env.car_info, road_info=sim_env.road_info)

    adv_policy = TanhGaussianPolicy(
        num_state,
        num_action_adv,
        arch=FLAGS.policy_arch,
        log_std_multiplier=FLAGS.policy_log_std_multiplier,
        log_std_offset=FLAGS.policy_log_std_offset,
        orthogonal_init=FLAGS.orthogonal_init,
    )

    qf1_adv = FullyConnectedQFunction(
        num_state,
        num_action_adv,
        arch=FLAGS.qf_arch,
        orthogonal_init=FLAGS.orthogonal_init,
        is_LN=True
    )
    target_qf1_adv = deepcopy(qf1_adv)

    qf2_adv = FullyConnectedQFunction(
        num_state,
        num_action_adv,
        arch=FLAGS.qf_arch,
        orthogonal_init=FLAGS.orthogonal_init,
        is_LN=True
    )
    target_qf2_adv = deepcopy(qf2_adv)

    if FLAGS.cql_adv.target_entropy >= 0.0:
        FLAGS.cql_adv.target_entropy = -np.prod(train_sampler.env.action_space_adv).item()

    cql_adv = ConservativeSAC(FLAGS.cql_adv, adv_policy, qf1_adv, qf2_adv, target_qf1_adv,
                              target_qf2_adv,  alpha_l2=FLAGS.alpha_l2)
    cql_adv.torch_to_device(FLAGS.device)

    sampler_adv_policy = SamplerPolicy(adv_policy, FLAGS.device)

    viskit_metrics = {}

    for l in range(FLAGS.n_loops):
        for epoch in trange(FLAGS.n_epochs_adv):
            metrics = {'epoch': epoch}

            # TODO: 采样数据，并通过判别器判别
            with Timer() as rollout_timer:
                # train_sampler.sample(
                #     adv_policy=sampler_adv_policy,
                #     n_steps=FLAGS.n_rollout_steps_per_epoch,
                #     deterministic=False, replay_buffer_ego=None, replay_buffer_adv=replay_buffer_tempt,
                #     joint_noise_std=FLAGS.joint_noise_std
                # )

                # TODO: 判别器
                # 1. 输入replay_buffer_tempt的数据，输出到replay_buffer_expend
                # 2. 清空replay_buffer_tempt
                # 3. 后续考虑对replay_buffer_expend再次判别
                metrics['epoch'] = epoch

            with Timer() as train_timer:
                # TODO: 增加对增广数据的采样；需不需要增加二者数据采样的比例？
                data = replay_buffer_real.sample_all()
                # data_expend = replay_buffer_expend.sample_all()
                # data = {key: torch.cat((data_expend.get(key, []), data_real.get(key, [])))
                #         for key in set(data_expend) | set(data_real)}
                for batch_idx in trange(FLAGS.n_train_step_per_epoch):
                    batch_adv = subsample_batch(data, FLAGS.batch_size)
                    # batch_adv = batch_to_torch(batch_adv, FLAGS.device)
                    if batch_idx + 1 == FLAGS.n_train_step_per_epoch:
                        metrics.update(
                            prefix_metrics(cql_adv.train(batch_adv, bc=epoch < FLAGS.bc_epochs), 'cql_adv')
                        )
                    else:
                        cql_adv.train(batch_adv, bc=epoch < FLAGS.bc_epochs)

            with Timer() as eval_timer:
                if epoch == 0 or (epoch + 1) % FLAGS.eval_period == 0:
                    trajs = eval_sampler.sample(
                        adv_policy=sampler_adv_policy,
                        n_trajs=FLAGS.eval_n_trajs, deterministic=True
                    )
                    metrics['average_return_adv'] = np.mean([np.mean(t['rewards_adv']) for t in trajs])
                    metrics['average_return_ego'] = np.mean([np.mean(t['rewards_ego']) for t in trajs])
                    metrics['average_traj_length'] = np.mean([len(t['rewards_adv']) for t in trajs])
                    metrics['metrics_av_crash'] = np.mean([t["metrics_av_crash"] for t in trajs])
                    metrics['metrics_bv_crash'] = np.mean([t["metrics_bv_crash"] for t in trajs])
                    metrics['ACT'] = 0 if metrics['metrics_av_crash'] == 0 else \
                        np.sum([t["collision_time"] for t in trajs]) / (metrics['metrics_av_crash'] * len(trajs))
                    metrics['ACD'] = 0 if metrics['metrics_av_crash'] == 0 else \
                        np.sum([t["collision_dis"] for t in trajs]) / (metrics['metrics_av_crash'] * len(trajs))
                    metrics['CPS'] = (metrics['metrics_av_crash'] * len(trajs)) / np.sum([t["traj_time"] for t in trajs])
                    metrics['CPM'] = (metrics['metrics_av_crash'] * len(trajs)) / np.sum([t["traj_dis"] for t in trajs])

                    if FLAGS.save_model and FLAGS.USED_wandb:
                        save_data = {'cql_adv': cql_adv,
                                     'variant': variant, 'epoch': epoch}
                        wandb_logger.save_pickle(save_data, 'model.pkl')

            metrics['rollout_time'] = rollout_timer()
            metrics['train_time'] = train_timer()
            metrics['eval_time'] = eval_timer()
            metrics['epoch_time'] = rollout_timer() + train_timer() + eval_timer()
            # metrics['epoch_time'] = train_timer() + eval_timer()
            if FLAGS.USED_wandb:
                wandb_logger.log(metrics)
            viskit_metrics.update(metrics)
            logger.record_dict(viskit_metrics)
            logger.dump_tabular(with_prefix=False, with_timestamp=False)

    if FLAGS.save_model and FLAGS.USED_wandb:
        save_data = {'cql_adv': cql_adv,
                     'variant': variant, 'epoch': epoch}
        wandb_logger.save_pickle(save_data, 'model.pkl')


if __name__ == '__main__':
    absl.app.run(main)
