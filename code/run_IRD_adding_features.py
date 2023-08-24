import time

print('importing')

start = time.time()
import datetime
print('time: '+str(datetime.datetime.now()))
import numpy as np
from inference_class import Inference
from gridworld import GridworldEnvironment, NStateMdpGaussianFeatures,\
    GridworldMdpWithDistanceFeatures, GridworldMdp
# from gridworldtfenv import GridWorldTFEnv, GWModel
from query_chooser_class import Experiment
from random import choice, seed
import copy
from utils import Distribution
import sys
import argparse
import tensorflow as tf
tf.compat.v1.disable_eager_execution()

print('Time to import: {deltat}'.format(deltat=time.time() - start))




def pprint(y):
    print(y)
    return y



# ==================================================================================================== #
# ==================================================================================================== #
if __name__=='__main__':
    parser = argparse.ArgumentParser()
    #parser.add_argument('-c','--c', action='append', required=True) # c for choosers
    parser.add_argument('--exp_name',type=str,default='no_exp_name')
    parser.add_argument('--size_true_space',type=int,default=1000000)
    parser.add_argument('--size_proxy_space',type=int,default=100)  # Sample subspace for exhaustive
    parser.add_argument('--seed',type=int,default=1)
    parser.add_argument('--beta',type=float,default=0.2)
    parser.add_argument('--num_states',type=int,default=100)  # 10 options if env changes over time, 100 otherwise
    parser.add_argument('--dist_scale',type=float,default=0.2) # test briefly to get ent down
    parser.add_argument('--height',type=int,default=12)
    parser.add_argument('--width',type=int,default=12)
    parser.add_argument('--mdp_type',type=str,default='gridworld')
    parser.add_argument('--test_samples',type=str,default=1000) # size for the sample of weigths on testing the variance
    parser.add_argument('--feature_dim',type=int,default=10)    # 10 if positions fixed, 100 otherwise
    parser.add_argument('--well_spec',type=int,default=1)    # default is well-specified
    parser.add_argument('--subsampling',type=int,default=1)
    parser.add_argument('--num_subsamples',type=int,default=10000) # default: 10000
    parser.add_argument('--weighting',type=int,default=1)
    parser.add_argument('--linear_features',type=int,default=1)
    parser.add_argument('--objective',type=str,default='entropy')
    parser.add_argument('--log_objective',type=int,default=1)
    parser.add_argument('--rational_test_planner',type=int,default=1)
    # args for experiment with correlated features
    parser.add_argument('--repeated_obj',type=int,default=0)  # Creates gridworld with k object types, k features, and num_objects >= k objects
    parser.add_argument('--num_obj_if_repeated',type=int,default=50)  # Usually feature_dim is # of objects except for correlated features experiment. Must be > feature_dim
    parser.add_argument('--decorrelate_test_feat',type=int,default=1)

    # args for optimization
    parser.add_argument('-weights_dist_init',type=str,default='normal2')
    parser.add_argument('-weights_dist_search',type=str,default='normal2')
    parser.add_argument('--lr',type=float,default=20)  # Learning rate
    parser.add_argument('--only_optim_biggest',type=int,default=1)
    parser.add_argument('--num_iters_optim',type=int,default=10) # gradient steps for optimizing the weights
    parser.add_argument('--beta_planner',type=float,default=0.5) # 1 for small version of results
    parser.add_argument('--num_queries_max',type=int,default=2000) # default: 2000
    parser.add_argument('--discretization_size',type=int,default=5) # for continuous query selection
    parser.add_argument('--discretization_size_human',type=int,default=5)   # for continuous query actually posed

    # args for testing full IRD
    parser.add_argument('--proxy_space_is_true_space', type=int, default=0)
    parser.add_argument('--full_IRD_subsample_belief', type=str, default='no')  # other options: yes, uniform

    # what we care about
    parser.add_argument('-c','--c',type=str,default='incremental_optimize')
    parser.add_argument('--visualize',type=int,default=1) # make visuals
    parser.add_argument('--write_csv',type=int,default=1) # write results to csv
    parser.add_argument('--num_trajectory',type=int,default=20) # number of points in trajectory in plots
    parser.add_argument('--num_experiments',type=int,default=1) # default: 5
    parser.add_argument('--num_regret_envs',type=int,default=10) # default: 10, number of test envs for computing the regrets
    parser.add_argument('--query_size',type=int,default=5) # default:5
    parser.add_argument('--num_batch',type=int,default=6) # number of batches, default: 5
    parser.add_argument('--num_iter',type=int,default=2) # number of queries asked, default:5, or 20 for AIRD
    parser.add_argument('--batch_size',type=int,default=5) # size of batch, default: 5
    parser.add_argument('--value_iters',type=int,default=15) # number of iterations in planner, max_reward / (1-gamma) or height+width
    parser.add_argument('--var_type',type=str,default='worst') # or 'worst' (default: 'sub')
    parser.add_argument('--var_coef',type=float,default=100.)
    parser.add_argument('--living_reward',type=float,default=0.0) # default -0.01
    parser.add_argument('--sample_size',type=int,default=100) # size for the sample of weigths from distribution
    parser.add_argument('--gamma',type=float,default=1.) # discount for q learning
    parser.add_argument('--alpha',type=float,default=1.) # q learning rate

    args = parser.parse_args()
    # print(args)
    # assert args.discretization_size % 2 == 1

    # Experiment description
    adapted_description = False
    # print "Adapted description: ", adapted_description

    # Set parameters
    dummy_rewards = np.zeros(args.feature_dim)
    choosers = [args.c]
    SEED = args.seed
    seed(SEED)
    np.random.seed(SEED)
    tf.compat.v1.set_random_seed(SEED)
    exp_name = args.exp_name
    beta = args.beta
    num_states = args.num_states
    size_reward_space_true = args.size_true_space
    size_reward_space_proxy = args.size_proxy_space
    batch_size = args.batch_size
    var_type = args.var_type
    var_coef = args.var_coef
    living_reward = args.living_reward
    sample_size = args.sample_size
    test_samples = args.test_samples
    num_batch = args.num_batch
    num_queries_max = args.num_queries_max
    num_experiments = args.num_experiments
    num_iter_per_experiment = args.num_iter #; print('num iter = {i}'.format(i=num_iter_per_experiment))
    # Params for Gridworld
    gamma = args.gamma
    query_size = args.query_size
    dist_scale = args.dist_scale
    height = args.height
    width = args.width
    num_iters_optim = args.num_iters_optim
    p_wall = 0.35 if args.height < 20 else 0.1

    # These will be in the folder name of the log
    exp_params = {
        'exp_name': exp_name,
        # 'rational_test_planner': args.rational_test_planner,
        'qsize': query_size,
        # 'size_true': size_reward_space_true,
        # 'size_proxy': size_reward_space_proxy,
        'num_iter': num_iter_per_experiment,
        'var_type': var_type,
        'var_coef': var_coef,
        'batch_size': batch_size,
        'num_batch': num_batch,
        'sample_size': sample_size,
        'living_reward': living_reward,
        # 'optim_big': args.only_optim_biggest,
        # 'rational_test': args.rational_test_planner
        # 'proxy_is_true': args.proxy_space_is_true_space,
        # 'full_IRD_subs': args.full_IRD_subsample_belief,
    }
    'Sample true rewards and reward spaces'
    reward_space_true = np.array(np.random.randint(-9, 10, size=[size_reward_space_true, args.feature_dim]), dtype=np.int16)
    if not args.well_spec:
        true_rewards = [np.random.randint(-9, 10, size=[args.feature_dim]) for _ in range(num_experiments)]
    else:
        true_rewards = [choice(reward_space_true) for _ in range(num_experiments)]
        if args.repeated_obj:
            # Set values of proxy and goal
            for i, reward in enumerate(true_rewards):
                for j in range(args.feature_dim):
                    if reward[j] > 7: reward[j] = np.random.randint(-9, 6)
                reward[-1] = 9
                reward[-2] = -2
                true_rewards[i] = reward
                reward_space_true[i,:] = reward
    prior_avg = -0.5 * np.ones(args.feature_dim) + 1e-4 * np.random.exponential(1,args.feature_dim) # post_avg for uniform prior + noise

    'Set up env and agent for NStateMdp'
    if args.mdp_type == 'bandits':
        'Create batch MDPs'
        #make num_batch inferences and each inference have batch_size mdp's and env's
        exp_inferences = []
        for j in range(num_experiments):
            reward_space_proxy = reward_space_true if args.proxy_space_is_true_space \
                else np.random.randint(-9, 10, size=[size_reward_space_proxy, args.feature_dim])
            inferences = []
            for l in range(num_batch):
                mdps, envs = [], []
                for k in range(batch_size):
                    mdp = NStateMdpGaussianFeatures(num_states=num_states, rewards=np.zeros(args.feature_dim), start_state=0, preterminal_states=[],
                                            feature_dim=args.feature_dim, num_states_reachable=num_states, SEED=SEED+i*50+100)
                    env = GridworldEnvironment(mdp)
                    mdps.append(mdp)
                    envs.append(env)
                inference = Inference(
                    mdps, envs, beta, reward_space_true, reward_space_proxy)
                inferences.append(inference)
            exp_inferences.append(inferences)


    # Set up env and agent for gridworld
    elif args.mdp_type == 'gridworld':
        'Create batch MDPs'
        #make num_batch inferences and each inference have batch_size mdp's and env's
        exp_inferences = []
        for j in range(num_experiments):
            reward_space_proxy = reward_space_true if args.proxy_space_is_true_space \
                else np.random.randint(-9, 10, size=[size_reward_space_proxy, args.feature_dim])
            inferences = []
            feat_add_rate = args.feature_dim/num_batch
            for l in range(num_batch):
                mdps, envs = [], []
                for k in range(batch_size):
                    grid, goals = GridworldMdp.generate_random(args,height,width,0.35,args.feature_dim,None,living_reward=living_reward, print_grid=False)
                    mdp = GridworldMdpWithDistanceFeatures(grid, goals, args, dist_scale, living_reward=living_reward, noise=0, var_coef=var_coef, var_type=var_type)
                    # delete the last features from mdp
                    # we want to add in total args.feature_dim dimensions in num_batch iterations
                    # we will add ceil(args.feature_dim/num_batch) features in each one
                    added_feat = int(np.ceil((l+1)*feat_add_rate))
                    mdp.feature_matrix[:,:,added_feat+1:] = 0
                    # print(mdp.feature_matrix[6][6])
                    env = GridworldEnvironment(mdp)
                    mdps.append(mdp)
                    envs.append(env)
                inference = Inference(
                    mdps, envs, beta, reward_space_true, reward_space_proxy)
                inferences.append(inference)
            exp_inferences.append(inferences)
        
        regret_mdps, regret_envs = [], []
        for m in range(args.num_regret_envs):
            grid, goals = GridworldMdp.generate_random(args,height,width,0.35,args.feature_dim,None,living_reward=living_reward, print_grid=False)
            mdp = GridworldMdpWithDistanceFeatures(grid, goals, args, dist_scale, living_reward=living_reward, noise=0, var_coef=var_coef, var_type=var_type)
            env = GridworldEnvironment(mdp)
            regret_mdps.append(mdp)
            regret_envs.append(env)

        regret_inf = Inference(regret_mdps, regret_envs, beta, reward_space_true, reward_space_proxy=[])


    else:
        raise ValueError('Unknown MDP type: ' + str(args.mdp_type))



    'Run experiment'
    def run_experiment(query_size, exp_inferences, true_rewards, prior_avg):
        experiment = Experiment(true_rewards, query_size, num_queries_max,
                                args, choosers, SEED, exp_params, exp_inferences, regret_inf, prior_avg)
        results = experiment.get_experiment_stats(num_iter_per_experiment, num_batch, num_experiments)


        print('__________________________Finished experiment__________________________')

    run_experiment(query_size, exp_inferences, true_rewards, prior_avg)
