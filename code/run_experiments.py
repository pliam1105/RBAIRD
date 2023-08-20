from subprocess import call

# Discrete experiments

NUM_EXPERIMENTS = 1  # Modify this to change the sample size

query_sizes = [2,3,5,10]
choosers = ['incremental_optimize',
            # 'greedy_discrete',
            # 'random',
            # 'exhaustive'
            ]
total_nums = {'incremental_optimize': 50, 'greedy_discrete': 0, 'random':50, 'full': 0}
reward_params = [{'var_type': 'sub', 'var_coef': 10, 'living_reward': -0.01,'sample_size': 100},
                 {'var_type': 'sub', 'var_coef': 100, 'living_reward': -0.01,'sample_size': 100},
                 {'var_type': 'sub', 'var_coef': 1000, 'living_reward': -0.01,'sample_size': 100},
                 {'var_type': 'worst', 'var_coef': 100, 'living_reward': -0.01,'sample_size': 10},
                 {'var_type': 'worst', 'var_coef': 100, 'living_reward': -0.01,'sample_size': 100},
                #  {'var_type': 'sub', 'var_coef': 100, 'living_reward': -1,'sample_size': 100},
                #  {'var_type': 'sub', 'var_coef': 100, 'living_reward': -10,'sample_size': 100},
                #  {'var_type': 'sub', 'var_coef': 100, 'living_reward': -100,'sample_size': 100},
                 {'var_type': 'worst', 'var_coef': 100, 'living_reward': -0.01,'sample_size': 1000}
                 ]

optimal_evolution_params = [
                {'chooser': 'incremental_optimize', 'query_size':5, 'num_batch': 4, 'num_iter': 5, 'batch_size': 5, # basic RBAIRD
                'var_type':'sub', 'var_coef': 100, 'living_reward': -0.01, 'sample_size': 100},
                {'chooser': 'incremental_optimize', 'query_size':5, 'num_batch': 4, 'num_iter': 3, 'batch_size': 10, # many environments on batch
                'var_type':'sub', 'var_coef': 100, 'living_reward': -0.01, 'sample_size': 100},
                {'chooser': 'incremental_optimize', 'query_size':5, 'num_batch': 11, 'num_iter': 1, 'batch_size': 5, # many batches
                'var_type':'sub', 'var_coef': 100, 'living_reward': -0.01, 'sample_size': 100},
                            ]
risk_evolution_params = [
                {'chooser': 'incremental_optimize', 'query_size':5, 'num_batch': 7, 'num_iter': 2, 'batch_size': 5,
                'var_type':'sub', 'var_coef': 100, 'living_reward': -0.01, 'sample_size': 100},
                {'chooser': 'incremental_optimize', 'query_size':5, 'num_batch': 7, 'num_iter': 2, 'batch_size': 5,
                'var_type':'worst', 'var_coef': 100, 'living_reward': -0.01, 'sample_size': 10},
                {'chooser': 'incremental_optimize', 'query_size':5, 'num_batch': 7, 'num_iter': 2, 'batch_size': 5,
                'var_type':'worst', 'var_coef': 100, 'living_reward': -0.01, 'sample_size': 100},
                        ]
risk_param_params = [
                {'chooser': 'incremental_optimize', 'query_size':5, 'num_batch': 4, 'num_iter': 5, 'batch_size': 5,
                'var_type':'sub', 'var_coef': 100, 'living_reward': -0.01, 'sample_size': 100},
                {'chooser': 'incremental_optimize', 'query_size':5, 'num_batch': 4, 'num_iter': 5, 'batch_size': 5,
                'var_type':'sub', 'var_coef': 1, 'living_reward': -0.01, 'sample_size': 100},
                {'chooser': 'incremental_optimize', 'query_size':5, 'num_batch': 4, 'num_iter': 5, 'batch_size': 5,
                'var_type':'worst', 'var_coef': 100, 'living_reward': -0.01, 'sample_size': 10},
                {'chooser': 'incremental_optimize', 'query_size':5, 'num_batch': 4, 'num_iter': 5, 'batch_size': 5,
                'var_type':'worst', 'var_coef': 100, 'living_reward': -0.01, 'sample_size': 100},
                {'chooser': 'incremental_optimize', 'query_size':5, 'num_batch': 4, 'num_iter': 5, 'batch_size': 5,
                'var_type':'sub', 'var_coef': 100, 'living_reward': -1.0, 'sample_size': 100},
                    ]
# choosers_continuous = ['feature_entropy_search_then_optim', 'feature_random', 'feature_entropy_random_init_none'] #'feature_entropy_init_none', 'feature_entropy_search']
# choosers_discrete = ['greedy_discrete', 'random', 'exhaustive', 'incremental_optimize']
# mdp_types = ['gridworld']
# num_iter = {'gridworld': '20', 'bandits': '20'}
# num_iter = 5
# num_subsamples_full = '5000'; num_subsamples_not_full = '5000'
# beta_both_mdps = '0.5'
# num_q_max = '100' # default: 10000
# rsize = '1000000'
# proxy_space_is_true_space = '0'
# exp_name = '14May_reward_hacking'

def run(chooser, exp_name, qsize, num_batch, num_iter, batch_size, var_type, var_coef, living_reward, sample_size):
    command = ['python3', 'run_IRD.py',
               '--exp_name', str(exp_name),
               '--query_size', str(qsize),
               '--num_experiments', str(NUM_EXPERIMENTS),
               '--num_batch', str(num_batch),
               '--num_iter', str(num_iter),
               '--batch_size', str(batch_size),
               '--var_type', var_type,
               '--var_coef', str(float(var_coef)),
               '--living_reward', str(float(living_reward)),
               '--sample_size', str(sample_size),
               '-c', chooser,]
    print('Running command', ' '.join(command))
    call(command)

def run_chooser(chooser):
    total_num = total_nums[chooser]
    for num_batch in [2,5,10,20]:
        for batch_size in [1,5,10,20]:
            if(num_batch*batch_size > total_num):
                continue
            num_iter = total_num // (num_batch*batch_size)
            qsize = 5
            for param in reward_params:
                run(chooser, 'full',qsize, num_batch, num_iter, batch_size, param['var_type'], param['var_coef'], param['living_reward'], param['sample_size'])

def run_optimal_evolution():
    for params in optimal_evolution_params:
        run(params['chooser'], 'optimal_evolution',params['query_size'], params['num_batch'], params['num_iter'], params['batch_size'], 
            params['var_type'], params['var_coef'], params['living_reward'], params['sample_size'])
        
def run_risk_evolution():
    for params in risk_evolution_params:
        run(params['chooser'], 'risk_evolution',params['query_size'], params['num_batch'], params['num_iter'], params['batch_size'], 
            params['var_type'], params['var_coef'], params['living_reward'], params['sample_size'])
        
def run_risk_params():
    for params in risk_param_params:
        run(params['chooser'], 'risk_params',params['query_size'], params['num_batch'], params['num_iter'], params['batch_size'], 
            params['var_type'], params['var_coef'], params['living_reward'], params['sample_size'])

if __name__ == '__main__':
    # for chooser in choosers:
    #     run_chooser(chooser)
    run_optimal_evolution()
    run_risk_evolution()
    run_risk_params()