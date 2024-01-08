from ruamel.yaml import YAML, dump, RoundTripDumper
from raisimGymTorch.env.RaisimGymVecEnv import RaisimGymVecEnv as VecEnv
from raisimGymTorch.helper.raisim_gym_helper import ConfigurationSaver, load_param, tensorboard_launcher
from raisimGymTorch.env.bin.prisma_walker import NormalSampler
from raisimGymTorch.env.bin.prisma_walker import RaisimGymEnv
import os
import math
import time
import raisimGymTorch.algo.ppo.module as ppo_module
import raisimGymTorch.algo.ppo.ppo as PPO
import torch.nn as nn
import numpy as np
import torch
import datetime
import argparse


# task specification
task_name = "prisma_walker_locomotion"

# configuration
parser = argparse.ArgumentParser()
parser.add_argument('-m', '--mode', help='set mode either train or test', type=str, default='train')
parser.add_argument('-w', '--weight', help='pre-trained weight path', type=str, default='')
args = parser.parse_args()
mode = args.mode
weight_path = args.weight

# check if gpu is available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# directories
task_path = os.path.dirname(os.path.realpath(__file__))
home_path = task_path + "/../../../../.."  #Path to raisim folder (where there is GymTorch, Unity etc)
# config
cfg = YAML().load(open(task_path + "/cfg.yaml", 'r'))

# create environment from the configuration file
env = VecEnv(RaisimGymEnv(home_path + "/rsc", dump(cfg['environment'], Dumper=RoundTripDumper)))
env.seed(cfg['seed'])

# shortcuts
ob_dim = env.num_obs  
act_dim = env.num_acts 

num_threads = cfg['environment']['num_threads']

# Training
n_steps = math.floor(cfg['environment']['max_time'] / cfg['environment']['control_dt'])
total_steps = n_steps * env.num_envs

avg_rewards = []

actor = ppo_module.Actor(ppo_module.MLP(cfg['architecture']['policy_net'], nn.LeakyReLU, ob_dim, act_dim),
                         ppo_module.MultivariateGaussianDiagonalCovariance(act_dim,
                                                                           env.num_envs,
                                                                           1.0,
                                                                           NormalSampler(act_dim),
                                                                           cfg['seed']),
                         device)
critic = ppo_module.Critic(ppo_module.MLP(cfg['architecture']['value_net'], nn.LeakyReLU, ob_dim, 1),
                           device)


#saver = ConfigurationSaver(log_dir=home_path + "/raisimGymTorch/data/"+task_name,
#                           save_items=[task_path + "/cfg.yaml", task_path + "/Environment.hpp", task_path + "/runner.py"])  

#tensorboard_launcher(saver.data_dir+"/..")  # press refresh (F5) after the first ppo update
#lam = [0.9, 0.91, 0.92, 0.93, 0.94, 0.95, 0.96, 0.97, 0.98, 0.99, 1]
lam = [0.965]
for i in lam:
    saver = ConfigurationSaver(log_dir=home_path + "/raisimGymTorch/data/"+task_name,
                           save_items=[task_path + "/cfg.yaml", task_path + "/Environment.hpp", task_path + "/runner.py"],
                           lam = i)  

    print("lambda = ", i)
    ppo = PPO.PPO(actor=actor,
                critic=critic,
                num_envs=cfg['environment']['num_envs'],
                num_transitions_per_env=n_steps,
                num_learning_epochs=3,  #4 in origine
                gamma=0.998,
                lam=i, #0.95
                num_mini_batches=4, #4 in origine
                device=device,
                log_dir=saver.data_dir,
                shuffle_batch=False,
                )

    if mode == 'retrain':
        load_param(weight_path, env, actor, critic, ppo.optimizer, saver.data_dir)
    best_rewards = -100
    record_flag = False
    first_reward = 0
    average_ll_performance = 0
    for update in range(15001):            
        start = time.time()
        env.reset()
        reward_ll_sum = 0
        done_sum = 0
        average_dones = 0.

        if  update % cfg['environment']['eval_every_n'] == 0 and update !=0:
            print("Visualizing and evaluating the current policy")
            torch.save({
                'actor_architecture_state_dict': actor.architecture.state_dict(),
                'actor_distribution_state_dict': actor.distribution.state_dict(),
                'critic_architecture_state_dict': critic.architecture.state_dict(),
                'optimizer_state_dict': ppo.optimizer.state_dict(),
            }, saver.data_dir+"/full_"+str(update)+'.pt')
            # we create another graph just to demonstrate the save/load method
            loaded_graph = ppo_module.MLP(cfg['architecture']['policy_net'], nn.LeakyReLU, ob_dim, act_dim)
            loaded_graph.load_state_dict(torch.load(saver.data_dir+"/full_"+str(update)+'.pt')['actor_architecture_state_dict'])

            env.turn_on_visualization()
            env.start_video_recording(datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S") + "policy_"+str(update)+'.mp4')
            #se commento questo for il robot non si muove
            for step in range(n_steps):
                with torch.no_grad(): #no gradient computation
                    frame_start = time.time()
                    obs = env.observe(False)
                    action_ll = loaded_graph.architecture(torch.from_numpy(obs).cpu())
                    reward_ll, dones = env.step(action_ll.cpu().detach().numpy()) #env.step fa muovere tutto
                    #print("azioni",action_ll.cpu().detach().numpy())
                    frame_end = time.time()
                    wait_time = cfg['environment']['control_dt'] - (frame_end-frame_start)
                    if wait_time > 0.:
                        time.sleep(wait_time)

            env.stop_video_recording()
            env.turn_off_visualization()

            env.reset()
            env.save_scaling(saver.data_dir, str(update))
            record_flag = False

        # actual training
        for step in range(n_steps): #n_steps 400
            obs = env.observe() #dim 120 (immagino sia (generalized coordinate+njoints)*num_envs)
            action = ppo.act(obs)
            #print(np.size(action))
            reward, dones = env.step(action)
            ppo.step(value_obs=obs, rews=reward, dones=dones)
            done_sum = done_sum + np.sum(dones)
            reward_ll_sum = reward_ll_sum + np.sum(reward)
        # take st step to get value obs
        obs = env.observe()
        #print(np.size(obs)) #120
        ppo.update(actor_obs=obs, value_obs=obs, log_this_iteration=update % 10 == 0, update=update)
        average_ll_performance = reward_ll_sum / total_steps
        average_dones = done_sum / total_steps
        avg_rewards.append(average_ll_performance)
        if update == 0:
            first_reward = average_ll_performance

        """if average_ll_performance < (first_reward -1) or (update >50 and average_ll_performance<first_reward):
            print("Fail")
            break"""

        if average_ll_performance > best_rewards:
            best_rewards = average_ll_performance

            torch.save({
                'actor_architecture_state_dict': actor.architecture.state_dict(),
                'actor_distribution_state_dict': actor.distribution.state_dict(),
                'critic_architecture_state_dict': critic.architecture.state_dict(),
                'optimizer_state_dict': ppo.optimizer.state_dict(),
            }, saver.data_dir+"/full_"+str(update)+'.pt')
            # we create another graph just to demonstrate the save/load method
  
            env.save_scaling(saver.data_dir, str(update))

        actor.update()
        actor.distribution.enforce_minimum_std((torch.ones(3)*0.2).to(device))

        # curriculum update. Implement it in Environment.hpp
        env.curriculum_callback()

        end = time.time()

        print('----------------------------------------------------')
        print('{:>6}th iteration'.format(update))
        print('{:<40} {:>6}'.format("average ll reward: ", '{:0.10f}'.format(average_ll_performance)))
        print('{:<40} {:>6}'.format("dones: ", '{:0.6f}'.format(average_dones)))
        print('{:<40} {:>6}'.format("time elapsed in this iteration: ", '{:6.4f}'.format(end - start)))
        print('{:<40} {:>6}'.format("fps: ", '{:6.0f}'.format(total_steps / (end - start))))
        print('{:<40} {:>6}'.format("real time factor: ", '{:6.0f}'.format(total_steps / (end - start)
                                                                        * cfg['environment']['control_dt'])))
        print('----------------------------------------------------\n')
