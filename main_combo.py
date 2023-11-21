import argparse
import os
import sys

import numpy as np
import torch

from models.combo.nets import MLP
from models.combo.modules import ActorProb
from models.combo.modules import Critic
from models.combo.modules import EnsembleDynamicsModel
from models.combo.modules import TanhDiagGaussian
from models.combo.ensemble_dynamics import EnsembleDynamics
from models.combo.nets import StandardScaler
from models.combo.nets import terminaltion_fn
# from models.combo.utils.load_dataset import qlearning_dataset
# from models.combo.buffer import ReplayBuffer
from buffer.replay_buffer import ReplayBuffer
# from models.combo.utils.logger import Logger, make_log_dirs
from models.combo.mb_policy_trainer import MBPolicyTrainer
from models.combo.policy.combo import COMBOPolicy
# from copy import deepcopy
import absl.app
import absl.flags
import wandb
from envs.envs import Env
# from buffer.sampler import TrajSampler
from utils.utils import define_flags_with_default, set_random_seed, get_user_flags
from utils.utils import WandBLogger
from utils.viskit.logging import logger, setup_logger
import datetime


"""
suggested hypers

halfcheetah-medium-v2: rollout-length=5, cql-weight=0.5
hopper-medium-v2: rollout-length=5, cql-weight=5.0
walker2d-medium-v2: rollout-length=1, cql-weight=5.0
halfcheetah-medium-replay-v2: rollout-length=5, cql-weight=0.5
hopper-medium-replay-v2: rollout-length=5, cql-weight=0.5
walker2d-medium-replay-v2: rollout-length=1, cql-weight=0.5
halfcheetah-medium-expert-v2: rollout-length=5, cql-weight=5.0
hopper-medium-expert-v2: rollout-length=5, cql-weight=5.0
walker2d-medium-expert-v2: rollout-length=1, cql-weight=5.0
"""

parser = argparse.ArgumentParser()
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--actor-lr", type=float, default=1e-4)
parser.add_argument("--critic-lr", type=float, default=3e-4)
parser.add_argument("--hidden-dims", type=int, nargs='*', default=[256, 256, 256])
parser.add_argument("--gamma", type=float, default=0.99)
parser.add_argument("--tau", type=float, default=0.005)
parser.add_argument("--alpha", type=float, default=0.2)
parser.add_argument("--auto-alpha", default=True)
parser.add_argument("--target-entropy", type=int, default=None)
parser.add_argument("--alpha-lr", type=float, default=1e-4)

parser.add_argument("--cql-weight", type=float, default=0.5)
parser.add_argument("--temperature", type=float, default=1.0)
parser.add_argument("--max-q-backup", type=bool, default=False)
parser.add_argument("--deterministic-backup", type=bool, default=True)
parser.add_argument("--with-lagrange", type=bool, default=False)
parser.add_argument("--lagrange-threshold", type=float, default=10.0)
parser.add_argument("--cql-alpha-lr", type=float, default=3e-4)
parser.add_argument("--num-repeat-actions", type=int, default=10)
parser.add_argument("--uniform-rollout", type=bool, default=False)
parser.add_argument("--rho-s", type=str, default="mix", choices=["model", "mix"])

parser.add_argument("--dynamics-lr", type=float, default=1e-3)
parser.add_argument("--dynamics-hidden-dims", type=int, nargs='*', default=[200, 200, 200, 200])
parser.add_argument("--dynamics-weight-decay", type=float, nargs='*', default=[2.5e-5, 5e-5, 7.5e-5, 7.5e-5, 1e-4])
parser.add_argument("--n-ensemble", type=int, default=7)
parser.add_argument("--n-elites", type=int, default=5)
parser.add_argument("--rollout-freq", type=int, default=1000)
parser.add_argument("--rollout-batch-size", type=int, default=50000)
parser.add_argument("--rollout-length", type=int, default=5)
parser.add_argument("--model-retain-epochs", type=int, default=5)
parser.add_argument("--real-ratio", type=float, default=0.5)
parser.add_argument("--load-dynamics-path", type=str, default=None)

parser.add_argument("--epoch", type=int, default=1000)
parser.add_argument("--step-per-epoch", type=int, default=1000)
parser.add_argument("--eval_episodes", type=int, default=10)
parser.add_argument("--batch-size", type=int, default=256)
parser.add_argument("--device", type=str, default="cuda:0" if torch.cuda.is_available() else "cpu")

parser.add_argument('--USED_wandb', type=str, default="False")
parser.add_argument('--ego_policy', type=str, default="sumo")  # "uniform", "sumo", "fvdm"
parser.add_argument('--adv_policy', type=str, default="RL")  # "uniform", "sumo", "fvdm"
parser.add_argument('--num_agents', type=int, default=1)
# parser.add_argument('--r_ego', type=str, default="r1")
parser.add_argument('--r_adv', type=str, default="r1")
parser.add_argument('--realdata_path', type=str, default="E:/scenario_generation/dataset/Re_2_H2O/r3_dis_10_car_2/")  #
# parser.add_argument('--realdata_path', type=str, default="../byH2O/dataset/r1_dis_10_car_2/")  #
parser.add_argument('--is_save', type=str, default="False")
parser.add_argument('--save_model', type=str, default="False")

args = parser.parse_args()
args.is_save = True if args.is_save == "True" else False
args.USED_wandb = True if args.USED_wandb == "True" else False
args.save_model = True if args.save_model == "True" else False
# print(args)
while len(sys.argv) > 1:
	sys.argv.pop()
FLAGS_DEF = define_flags_with_default(
#     USED_wandb = args.USED_wandb,
#     ego_policy=args.ego_policy,  # "uniform", "sumo", "fvdm"
#     adv_policy=args.adv_policy,  # "uniform", "sumo", "fvdm"
#     num_agents=args.num_agents,
#     # r_ego=args.r_ego,
#     r_adv=args.r_adv,
#     realdata_path=args.realdata_path,
#     # batch_ratio=args.batch_ratio,  # 仿真数据的比例
#     is_save=args.is_save,
#     device=args.device,
#     # cql_min_q_weight=args.cql_min_q_weight,
#     seed=args.seed,
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
    # n_epochs_ego=0,
    # n_epochs_adv=1000,
    n_loops=1,
    # bc_epochs=0,
    # n_rollout_steps_per_epoch=1000,
    # n_train_step_per_epoch=1000,
    eval_period=10,
    eval_n_trajs=20,

    # sac_ego=SAC.get_default_config(),
    # sac_adv=ConservativeSAC.get_default_config(),
    logging=WandBLogger.get_default_config(),
)



def main(argv):
    FLAGS = absl.flags.FLAGS
    run_name = f"COMBO_av={args.ego_policy}_" \
               f"bv={args.num_agents}-{args.adv_policy}_" \
               f"r-adv={args.r_adv}_" \
               f"seed={args.seed}_time={FLAGS.current_time}"
    if args.is_save:
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
    if args.USED_wandb:
        variant = get_user_flags(FLAGS, FLAGS_DEF)
        wandb_logger = WandBLogger(config=FLAGS.logging, variant=variant)
        wandb.run.name = run_name

        setup_logger(
            variant=variant,
            exp_id=wandb_logger.experiment_id,
            seed=args.seed,
            base_log_dir=FLAGS.logging.output_dir,
            include_exp_prefix_sub_dir=False
        )

    set_random_seed(args.seed)

    # create env and dataset
    env = Env(realdata_path=args.realdata_path, num_agents=args.num_agents, sim_horizon=FLAGS.max_traj_length,
              ego_policy=args.ego_policy, adv_policy=args.adv_policy,
              r_adv=args.r_adv, sim_seed=args.seed)
    # eval_sampler = TrajSampler(env, rootsavepath=eval_savepath, max_traj_length=FLAGS.max_traj_length)

    # dataset = qlearning_dataset(env)
    obs_shape = env.state_space[0]
    action_dim = env.action_space_adv[0]
    # max_action = env.action_space.high[0]

    # seed
    torch.backends.cudnn.deterministic = True

    # create policy model
    actor_backbone = MLP(input_dim=obs_shape, hidden_dims=args.hidden_dims)
    critic1_backbone = MLP(input_dim=obs_shape + action_dim, hidden_dims=args.hidden_dims)
    critic2_backbone = MLP(input_dim=obs_shape + action_dim, hidden_dims=args.hidden_dims)
    dist = TanhDiagGaussian(
        latent_dim=getattr(actor_backbone, "output_dim"),
        output_dim=action_dim,
        unbounded=True,
        conditioned_sigma=True
    )
    actor = ActorProb(actor_backbone, dist, args.device)
    critic1 = Critic(critic1_backbone, args.device)
    critic2 = Critic(critic2_backbone, args.device)
    actor_optim = torch.optim.Adam(actor.parameters(), lr=args.actor_lr)
    critic1_optim = torch.optim.Adam(critic1.parameters(), lr=args.critic_lr)
    critic2_optim = torch.optim.Adam(critic2.parameters(), lr=args.critic_lr)

    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(actor_optim, args.epoch)

    if args.auto_alpha:
        target_entropy = args.target_entropy if args.target_entropy \
            else -np.prod(env.action_space_adv)
        args.target_entropy = target_entropy
        log_alpha = torch.zeros(1, requires_grad=True, device=args.device)
        alpha_optim = torch.optim.Adam([log_alpha], lr=args.alpha_lr)
        alpha = (target_entropy, log_alpha, alpha_optim)
    else:
        alpha = args.alpha

    # create dynamics
    load_dynamics_model = True if args.load_dynamics_path else False
    dynamics_model = EnsembleDynamicsModel(
        obs_dim=obs_shape,
        action_dim=action_dim,
        hidden_dims=args.dynamics_hidden_dims,
        num_ensemble=args.n_ensemble,
        num_elites=args.n_elites,
        weight_decays=args.dynamics_weight_decay,
        device=args.device
    )
    dynamics_optim = torch.optim.Adam(
        dynamics_model.parameters(),
        lr=args.dynamics_lr
    )
    scaler = StandardScaler()
    termination_fn = terminaltion_fn
    dynamics = EnsembleDynamics(
        dynamics_model,
        dynamics_optim,
        scaler,
        termination_fn
    )

    if args.load_dynamics_path:
        dynamics.load(args.load_dynamics_path)

    # create policy
    policy = COMBOPolicy(
        dynamics,
        actor,
        critic1,
        critic2,
        actor_optim,
        critic1_optim,
        critic2_optim,
        action_space=action_dim,
        tau=args.tau,
        gamma=args.gamma,
        alpha=alpha,
        cql_weight=args.cql_weight,
        temperature=args.temperature,
        max_q_backup=args.max_q_backup,
        deterministic_backup=args.deterministic_backup,
        with_lagrange=args.with_lagrange,
        lagrange_threshold=args.lagrange_threshold,
        cql_alpha_lr=args.cql_alpha_lr,
        num_repeart_actions=args.num_repeat_actions,
        uniform_rollout=args.uniform_rollout,
        rho_s=args.rho_s
    )

    # create buffer
    fake_buffer = ReplayBuffer(obs_shape, action_dim, FLAGS.replay_buffer_size, device=args.device,
                               datapath=None)
    real_buffer = ReplayBuffer(obs_shape, action_dim, FLAGS.replay_buffer_size, device=args.device,
                               datapath=args.realdata_path)
    # real_buffer = ReplayBuffer(
    #     buffer_size=len(dataset["observations"]),
    #     obs_shape=obs_shape,
    #     obs_dtype=np.float32,
    #     action_dim=action_dim,
    #     action_dtype=np.float32,
    #     device=args.device
    # )
    # real_buffer.load_dataset(dataset)
    # fake_buffer = ReplayBuffer(
    #     buffer_size=args.rollout_batch_size * args.rollout_length * args.model_retain_epochs,
    #     obs_shape=obs_shape,
    #     obs_dtype=np.float32,
    #     action_dim=action_dim,
    #     action_dtype=np.float32,
    #     device=args.device
    # )

    # log
    # log_dirs = make_log_dirs(args.task, args.algo_name, args.seed, vars(args))
    # key: output file name, value: output handler type
    # output_config = {
    #     "consoleout_backup": "stdout",
    #     "policy_training_progress": "csv",
    #     "dynamics_training_progress": "csv",
    #     "tb": "tensorboard"
    # }
    # logger = Logger(log_dirs, output_config)
    # logger.log_hyperparameters(vars(args))

    # create policy trainer
    policy_trainer = MBPolicyTrainer(
        policy=policy,
        eval_env=env,
        real_buffer=real_buffer,
        fake_buffer=fake_buffer,
        logger=logger,
        rollout_setting=(args.rollout_freq, args.rollout_batch_size, args.rollout_length),
        epoch=args.epoch,
        step_per_epoch=args.step_per_epoch,
        batch_size=args.batch_size,
        real_ratio=args.real_ratio,
        eval_episodes=args.eval_episodes,
        lr_scheduler=lr_scheduler
    )

    # train
    if not load_dynamics_model:
        dynamics.train(real_buffer.sample_all(), logger, max_epochs_since_update=5)

    policy_trainer.train()


if __name__ == '__main__':
    absl.app.run(main)
