import numpy as np
import tensorflow as tf
tf.compat.v1.disable_eager_execution()
from itertools import product

from gridworld import Direction

class Model(object):
    def __init__(self, batch_size, feature_dim, gamma, query_size, discretization_size,
                 true_reward_space_size, num_unknown, beta, beta_planner,
                 objective, lr, discrete, optimize, args):
        self.initialized = False
        self.batch_size = batch_size
        self.feature_dim = feature_dim
        self.gamma = gamma
        self.query_size = query_size
        self.true_reward_space_size = true_reward_space_size
        self.beta = beta
        self.beta_planner = beta_planner
        self.lr = lr
        self.discrete = discrete
        self.optimize = optimize
        self.args = args
        # self.run_build_planner = False
        # self.run_build_risk_planner = False
        if discrete:
            self.K = query_size
            if optimize:
                self.num_unknown = num_unknown
        else:
            assert query_size <= 5
            num_posneg_vals = (discretization_size // 2)
            const = 9 // num_posneg_vals
            f_range = list(range(-num_posneg_vals * const, 10, const))
            print('Using', f_range, 'to discretize the feature')
            if len(f_range) != discretization_size:
                print('discretization size is off by ' + str(len(f_range) - discretization_size))
            # proxy_space = np.random.randint(-4,3,size=[30 * query_size, query_size])
            self.proxy_reward_space = list(product(f_range, repeat=query_size))
            self.K = len(self.proxy_reward_space)
        self.build_tf_graph(objective)

    def initialize(self, sess):
        if not self.initialized:
            self.initialized = True
            sess.run(self.initialize_op)

    def build_tf_graph(self, objective):
        self.name_to_op = {}
        self.build_weights()
        self.build_planner()
        self.build_map_to_posterior()
        # self.build_risk_planner()
        self.build_risk_q_planner()
        self.build_non_risk_q_planner()
        self.build_map_to_objective(objective)
        # Initializing the variables
        self.initialize_op = tf.compat.v1.global_variables_initializer()

    def build_weights(self):
        if self.discrete and self.optimize:
            self.build_discrete_weights_for_optimization()
        elif self.discrete:
            self.build_discrete_weights()
        else:
            self.build_continuous_weights()

    def build_discrete_weights_for_optimization(self):
        K, N = self.K, self.num_unknown
        dim = self.feature_dim
        self.weights_to_train = tf.Variable(
            tf.zeros([N, dim]), name="weights_to_train")
        self.weight_inputs = tf.compat.v1.placeholder(
            tf.float32, shape=[N, dim], name="weight_inputs")
        self.assign_op = self.weights_to_train.assign(self.weight_inputs)

        if N < K:
            self.known_weights = tf.compat.v1.placeholder(
                tf.float32, shape=[K - N, dim], name="known_weights")
            self.weights = tf.concat(
                [self.known_weights, self.weights_to_train], axis=0, name="weights")
        else:
            self.weights = self.weights_to_train

        self.name_to_op['weights'] = self.weights
        self.name_to_op['weights_to_train'] = self.weights_to_train

    def build_discrete_weights(self):
        self.weights = tf.compat.v1.placeholder(
            tf.float32, shape=[self.K, self.feature_dim], name="weights")

    def build_continuous_weights(self):
        query_size, dim, K = self.query_size, self.feature_dim, self.K
        num_fixed = dim - query_size
        self.query_weights= tf.constant(
            self.proxy_reward_space, dtype=tf.float32, name="query_weights")

        # if self.optimize:
        weight_inits = tf.random.normal([num_fixed], stddev=2)
        self.weights_to_train = tf.Variable(
            weight_inits, name="weights_to_train")
        self.weight_inputs = tf.compat.v1.placeholder(
            tf.float32, shape=[num_fixed], name="weight_inputs")
        self.assign_op = self.weights_to_train.assign(self.weight_inputs)
        self.fixed_weights = self.weights_to_train
        self.name_to_op['weights_to_train'] = self.weights_to_train
        self.name_to_op['weights_to_train[:3]'] = self.weights_to_train[:3]
        # else:
        #     self.fixed_weights = tf.constant(
        #         np.zeros([num_fixed], dtype=np.float32))

        # Let's say query is [1, 3] and there are 6 features.
        # query_weights = [10, 11] and weight_inputs = [12, 13, 14, 15].
        # Then we want self.weights to be [12, 10, 13, 11, 14, 15].
        # Concatenate to get [10, 11, 12, 13, 14, 15]
        repeated_weights = tf.stack([self.fixed_weights] * K, axis=0)
        unordered_weights = tf.concat(
            [self.query_weights, repeated_weights], axis=1)
        # Then permute using gather to get the desired result.
        # The permutation can be computed from the query [1, 3] using
        # get_permutation_from_query.
        self.permutation = tf.compat.v1.placeholder(tf.int32, shape=[dim])
        self.weights = tf.gather(unordered_weights, self.permutation, axis=-1)

        self.name_to_op['weights'] = self.weights
        self.name_to_op['query_weights'] = self.query_weights


    def build_planner(self):
        raise NotImplemented('Should be implemented in subclass')

    def build_risk_planner(self):
        raise NotImplemented('Should be implemented in subclass')
    
    def build_risk_q_planner(self):
        raise NotImplemented('Should be implemented in subclass')
    
    def build_non_risk_q_planner(self):
        raise NotImplemented('Should be implemented in subclass')

    def build_map_to_posterior(self):
        """
        Maps self.feature_exp (created by planner) to self.log_posterior.
        """
        # Get log likelihoods for true reward matrix
        true_reward_space_size = self.true_reward_space_size
        dim = self.feature_dim
        self.true_reward_matrix = tf.compat.v1.placeholder(
            tf.float32, [true_reward_space_size, dim], name="true_reward_matrix")
        self.log_true_reward_matrix = tf.math.log(self.true_reward_matrix, name='log_true_reward_matrix')


        # TODO: Inefficient to recompute this matrix on every forward pass.
        # We can cache it and feed in true reward indeces instead of true_reward_matrix. The matrix multiplication has
        # size_proxy x size_true x feature_dim complexity. The other calculations in this map have a factor feature_dim
        # less. However, storing this matrix takes size_proxy / feature_dim more memory. That's good for large feature_dim.
        self.avg_reward_matrix = tf.tensordot(
            self.feature_expectations, self.true_reward_matrix, axes=[-1, -1], name='avg_reward_matrix')

        log_likelihoods_new = self.beta * self.avg_reward_matrix


        # Calculate posterior
        # self.prior = tf.placeholder(tf.float32, name="prior", shape=(true_reward_space_size))
        self.log_prior = tf.compat.v1.placeholder(tf.float32, name="log_prior", shape=(true_reward_space_size))
        log_Z_w = tf.reduce_logsumexp(log_likelihoods_new, axis=0, name='log_Z_w')
        log_P_q_z = log_likelihoods_new - log_Z_w   # broadcasting
        # self.log_Z_q, max_a, max_b = logdot(log_P_q_z, tf.log(self.prior))
        self.log_Z_q = tf.reduce_logsumexp(log_P_q_z + self.log_prior, axis=1, name='log_Z_q', keepdims=True)
        # TODO: For BALD objective, just take entropy of Z_q - prior expected entropy of q
        # self.log_posterior = log_P_q_z + tf.log(self.prior) - self.log_Z_q
        self.log_posterior = log_P_q_z + self.log_prior - self.log_Z_q  # 2x broadcasting
        self.posterior = tf.exp(self.log_posterior, name="posterior")

        self.post_sum_to_1 = tf.reduce_sum(tf.exp(self.log_posterior), axis=1, name='post_sum_to_1')


        # Get log likelihoods for actual true reward
        self.true_reward = tf.compat.v1.placeholder(
            tf.float32, shape=[dim], name="true_reward")
        self.true_reward_tensor = tf.expand_dims(
            self.true_reward, axis=0, name="true_reward_tensor")
        self.avg_true_rewards = tf.tensordot(
            self.feature_expectations, tf.transpose(self.true_reward_tensor), axes=[-1, -2], name='avg_true_rewards')
        true_log_likelihoods = self.beta * self.avg_true_rewards
        log_true_Z_w = tf.reduce_logsumexp(true_log_likelihoods, axis=0, name='log_true_Z_w')
        self.log_true_answer_probs = true_log_likelihoods - log_true_Z_w

        self.name_to_op['true_reward_tensor'] = self.true_reward_tensor
        self.name_to_op['avg_true_rewards'] = self.avg_true_rewards
        self.name_to_op['true_log_likelihoods'] = true_log_likelihoods
        self.name_to_op['true_log_answer_probs'] = self.log_true_answer_probs
        self.name_to_op['log_true_Z_w'] = log_true_Z_w
        self.name_to_op['log_true_answer_probs'] = self.log_true_answer_probs

        # # Sample answer
        self.log_true_answer_probs = tf.reshape(self.log_true_answer_probs, shape=[1, -1])
        sample = tf.random.categorical(self.log_true_answer_probs, num_samples=1)
        sample = sample[0][0]
        self.true_log_posterior = self.log_posterior[sample]
        self.true_posterior = self.posterior[sample]

        self.name_to_op['sample'] = sample
        self.name_to_op['true_posterior'] = self.true_posterior
        self.name_to_op['true_log_posterior'] = self.true_log_posterior
        self.name_to_op['probs'] = tf.exp(self.log_true_answer_probs)

        # Get true posterior entropy
        scaled_log_true_posterior = self.true_log_posterior - 0.0001
        interm_tensor = scaled_log_true_posterior + tf.math.log(- scaled_log_true_posterior)
        self.true_ent = tf.exp(tf.reduce_logsumexp(
            interm_tensor, axis=0, name="true_entropy", keepdims=True))
        self.name_to_op['true_entropy'] = self.true_ent

        # Get true posterior_avg
        ## Not in log space
        self.post_weighted_true_reward_matrix = tf.multiply(self.true_posterior, tf.transpose(self.true_reward_matrix))
        self.true_post_avg = tf.reduce_sum(self.post_weighted_true_reward_matrix, axis=1, name='post_avg', keepdims=False)

        ## In log space (necessary?)
        # log_true_posterior_times_true_reward = self.true_log_posterior + tf.transpose(self.log_true_reward_matrix) # TODO: log true posteriors are log of negative
        # self.log_post_avg = tf.reduce_logsumexp(log_true_posterior_times_true_reward, axis=1, keep_dims=False)
        # self.name_to_op['log_post_avg'] = self.log_post_avg
        # self.post_avg = tf.exp(self.log_post_avg, name='post_avg')


        # Fill name to ops dict
        self.name_to_op['post_avg'] = self.true_post_avg
        self.name_to_op['avg_reward_matrix'] = self.avg_reward_matrix
        self.name_to_op['true_reward_matrix'] = self.true_reward_matrix
        # self.name_to_op['prior'] = self.prior
        self.name_to_op['log_prior'] = self.log_prior
        self.name_to_op['posterior'] = self.posterior
        self.name_to_op['log_posterior'] = self.log_posterior
        self.name_to_op['post_sum_to_1'] = self.post_sum_to_1


    def build_map_to_objective(self, objective):
        """
        :param objective: string that specifies the objective function
        """
        if 'entropy' == objective:
            # # Calculate exp entropy without log space trick
            # post_ent = - tf.reduce_sum(
            #     tf.multiply(tf.exp(self.log_posterior), self.log_posterior), axis=1, keep_dims=True, name='post_ent')
            # self.exp_post_ent = tf.reduce_sum(
            #     tf.multiply(post_ent, tf.exp(self.log_Z_q)), axis=0, keep_dims=True, name='exp_post_entropy')
            # self.name_to_op['entropy'] = self.exp_post_ent


            # Calculate entropy as exp logsumexp (log p + log (-log p))
            scaled_log_posterior = self.log_posterior - 0.0001
            interm_tensor = scaled_log_posterior + tf.math.log(- scaled_log_posterior)
            self.log_post_ent_new = tf.reduce_logsumexp(
                interm_tensor, axis=1, name="log_entropy_per_answer", keepdims=True)
            self.post_ent_new = tf.exp(self.log_post_ent_new)
            self.name_to_op['entropy_per_answer'] = self.post_ent_new
            self.log_exp_post_ent = tf.reduce_logsumexp(
                self.log_post_ent_new + self.log_Z_q, axis=0, keepdims=True, name='log_entropy')
            self.exp_post_ent = tf.exp(self.log_exp_post_ent)
            self.name_to_op['entropy'] = self.exp_post_ent

            if self.args.log_objective:
                self.objective = self.log_exp_post_ent
            else:
                self.objective = self.exp_post_ent

        if 'query_neg_entropy' == objective:
            scaled_log_answer_probs = self.log_Z_q - 0.0001
            interm_tensor = scaled_log_answer_probs + tf.math.log(- scaled_log_answer_probs)
            self.log_query_entropy = tf.reduce_logsumexp(
                interm_tensor, axis=-2, name='log_query_entropy', keepdims=True)

            self.query_entropy = tf.exp(self.log_query_entropy, name='query_entropy')
            self.name_to_op['log_query_entropy'] = self.log_query_entropy
            self.name_to_op['query_neg_entropy'] = -self.query_entropy

            if self.args.log_objective:
                self.objective = -self.log_query_entropy
            else:
                self.objective = -self.query_entropy

        if 'total_variation' == objective:

            self.post_averages, self.post_var = tf.nn.weighted_moments(
                self.true_reward_matrix, [1, 1], tf.stack([self.posterior] * self.feature_dim, axis=2),
                name="moments", keepdims=False)
            self.name_to_op['post_var'] = self.post_var

            self.total_variations = tf.reduce_sum(self.post_var, axis=-1, keepdims=False)
            self.name_to_op['total_variations'] = self.total_variations
            self.total_variations, self.log_Z_q = tf.reshape(self.total_variations, [-1]), tf.reshape(self.log_Z_q,[-1])
            self.total_variation = tf.tensordot(
                self.total_variations, tf.exp(self.log_Z_q), axes=[-1,-1] ,name='total_var')
            self.total_variation = tf.reshape(self.total_variation, shape=[1,1,-1])
            self.name_to_op['total_variation'] = self.total_variation

            if self.args.log_objective:
                self.objective = tf.math.log(self.total_variation)
            else:
                self.objective = self.total_variation

        # Set up optimizer
        if self.optimize:
            # optimizer = tf.train.AdamOptimizer(learning_rate=self.lr) # Make sure the momentum is reset for each model call
            self.lr_tensor = tf.constant(self.lr)
            self.optimizer = tf.compat.v1.train.GradientDescentOptimizer(learning_rate=self.lr_tensor)
            self.gradients, self.vs = list(zip(*self.optimizer.compute_gradients(self.objective)))
            # self.gradient_norm = tf.norm(tf.stack(gradients, axis=0))
            self.train_op = self.optimizer.apply_gradients(list(zip(self.gradients, self.vs)))
            self.name_to_op['gradients'] = self.gradients[-1]
            self.name_to_op['gradients[:4]'] = self.gradients[-1][:4]
            # self.name_to_op['gradient_norm'] = self.gradient_norm
            self.name_to_op['minimize'] = self.train_op
            self.name_to_op['lr_tensor'] = self.lr_tensor


    def compute(self, outputs, sess, mdp, query=None, log_prior=None, weight_inits=None, feature_expectations_input=None,
                gradient_steps=0, gradient_logging_outputs=[], true_reward=None, true_reward_matrix=None, lr=None):
        """
        Takes gradient steps to set the non-query features to the values that
        best optimize the objective. After optimization, calculates the values
        specified in outputs and returns them.

        :param outputs: List of strings, each specifying a value to compute.
        :param sess: tf.Session() object.
        :param mdp: The MDP whose true reward function we want to identify.
        :param query: List of features (integers) to ask the user to set.
        :param weight_inits: Initialization for the non-query features.
        :param gradient_steps: Number of gradient steps to take.
        :return: List of the same length as parameter `outputs`.
        """
        if weight_inits is not None:
            fd = {self.weight_inputs: weight_inits}
            sess.run([self.assign_op], feed_dict=fd)

        fd = {}
        self.update_feed_dict_with_mdp(mdp, fd)
        if feature_expectations_input is not None:
            fd[self.feature_expectations] = feature_expectations_input

        if log_prior is not None:
            fd[self.log_prior] = log_prior

        if query:
            if self.discrete and self.optimize:
                fd[self.known_weights] = query
            elif self.discrete:
                fd[self.weights] = query
            else:
                fd[self.permutation] = self.get_permutation_from_query(query)

        if true_reward is not None:
            fd[self.true_reward] = true_reward
        if true_reward_matrix is not None:
            fd[self.true_reward_matrix] = true_reward_matrix

        def get_op(name):
            if name not in self.name_to_op:
                raise ValueError("Unknown op name: " + str(name))
            return self.name_to_op[name]

        if gradient_steps > 0:
            # print("Reached gradient steps")
            if lr is not None:
                fd[self.lr_tensor] = lr
            ops = [get_op(name) for name in gradient_logging_outputs]
            other_ops = [self.train_op]
            for step in range(gradient_steps):
                # print("to run gd step: ",step)
                results = sess.run(ops + other_ops, feed_dict=fd)
                # print("ran gd step: ",step)
                # if ops and step % 1 == 0:
                #     print('Gradient step {0}: {1}'.format(step, results[:-1]))

        # print("Run planner: ",self.run_build_planner)
        # print("Run risk planner: ",self.run_build_risk_planner)
        return sess.run([get_op(name) for name in outputs], feed_dict=fd)

    def compute_batch(self, outputs, sess, mdps, query=None, log_prior=None, weight_inits=None, feature_expectations_inputs=None,
                gradient_steps=0, gradient_logging_outputs=[], true_reward=None, true_reward_matrix=None, lr=None):
        # +1 dimension: (all others are the same dimension) mpds, feature_expectations_input
        batch_size = self.batch_size
        if(feature_expectations_inputs == None): feature_expectations_inputs = [None]*batch_size
        if(mdps == None): mdps = [None]*batch_size
        # these two are changed and shared between models
        # feature_exp_out = []
        # risk_feature_exp_out = []
        # risk_var_out = []
        weights = weight_inits
        lg_prior = log_prior
        keep_array = ['feature_exps', 'risk_feature_exps', 'non_risk_feature_exps', 'risk_variance', 'policy', 'risk_policy', 'non_risk_policy', 'features', 'grid']
        met_out = {}
        for met in keep_array:
            met_out[met] = []
        for i in range(batch_size):
            mdp = mdps[i]
            ft_input = feature_expectations_inputs[i]
            # print(ft_input)
            if(ft_input is not None): ft_input = ft_input.eval(session=sess)
            out = self.compute(outputs, sess, mdp, query, lg_prior, weights, ft_input, 
                                    gradient_steps, gradient_logging_outputs, true_reward, true_reward_matrix, lr)
            if(i == batch_size-1):
                # we need the outputs from each batch, not the final one
                for metric in keep_array:
                    if (metric in outputs):
                        met_id = outputs.index(metric)
                        met_out[metric].append(out[met_id])
                        # print(str(batch_size),' ',len(met_out[metric]))
                        met_out[metric] = np.asarray(met_out[metric])
                        met_out[metric] = tf.convert_to_tensor(met_out[metric])
                        out[met_id] = met_out[metric]
                return out
            else:
                # we need weights and lg_prior in order to compute the next ones 
                if('weights_to_train' in outputs):
                    wt_t_id = outputs.index('weights_to_train')
                    weights = out[wt_t_id]
                if('true_log_posterior' in outputs):
                    t_l_p_id = outputs.index('true_log_posterior')
                    lg_prior = out[t_l_p_id]
                # keep all of batch
                for metric in keep_array:
                    if (metric in outputs):
                        met_id = outputs.index(metric)
                        met_out[metric].append(out[met_id])

    
    def update_feed_dict_with_mdp(self, mdp, fd):
        raise NotImplemented('Should be implemented in subclass')

    def get_permutation_from_query(self, query):
        dim = self.feature_dim
        # Running example: query = [1, 3], and we want indexes that will permute
        # weights [10, 11, 12, 13, 14, 15] to [12, 10, 13, 11, 14, 15].
        # Compute the feature numbers for unordered_weights.
        # This can be thought of as an unordered_weight -> feature map
        # In our example, this would be [1, 3, 0, 2, 4, 5]
        feature_order = query[:]
        for i in range(dim):
            if i not in feature_order:
                feature_order.append(i)
        # Invert the previous map to get the feature -> unordered_weight map.
        # This gives us [2, 0, 3, 1, 4, 5]
        indexes = [None] * dim
        for i in range(dim):
            indexes[feature_order[i]] = i
        return indexes


class BanditsModel(Model):

    def build_planner(self):
        self.features = tf.compat.v1.placeholder(
            tf.float32, name="features", shape=[None, self.feature_dim])
        self.name_to_op['features'] = self.features

        # Calculate state probabilities
        weights_expand = tf.expand_dims(self.weights,axis=1)
        intermediate_tensor = tf.multiply(tf.stack([self.features]*self.K,axis=0), weights_expand)
        self.reward_per_state = tf.reduce_sum(intermediate_tensor, axis=-1, keepdims=False, name="rewards_per_state")
        self.name_to_op['reward_per_state'] = self.reward_per_state
        self.name_to_op['q_values'] = self.reward_per_state

        # Rational planner
        if self.beta_planner == 'inf':
            self.best_state = tf.argmax(self.reward_per_state, axis=-1)
            self.num_states = tf.shape(self.reward_per_state)[1]
            self.state_probs = tf.one_hot(self.best_state, self.num_states)
        # Boltzmann rational planner
        else:
            self.state_probs = tf.nn.softmax(self.beta_planner * self.reward_per_state, axis=-1, name="state_probs")

        self.name_to_op['state_probs'] = self.state_probs
        self.name_to_op['state_probs_cut'] = self.state_probs[:5]
        """Changes: remove [0] from best state; remove reshape; stack probs on axis 2 (was 1); stack features on axis 1 (was 2); sum features on axis 1 (was 0); remove transpose."""

        # Calculate feature expectations
        probs_stack = tf.stack([self.state_probs] * self.feature_dim, axis=2)
        features_stack = tf.multiply(tf.stack([self.features] * self.K, axis=0), probs_stack, name='multi')
        self.feature_expectations = tf.reduce_sum(features_stack, axis=1, keepdims=False, name="feature_exps")
        self.name_to_op['feature_exps'] = self.feature_expectations


    def update_feed_dict_with_mdp(self, mdp, fd):
        fd[self.features] = mdp.convert_to_numpy_input()


class GridworldModel(Model):
    def __init__(self, batch_size, feature_dim, gamma, query_size, discretization_const,
                 true_reward_space_size, num_unknown, beta, beta_planner,
                 objective, lr, discrete, optimize, height, width, num_iters, args):
        self.height = height
        self.width = width
        self.num_iters = num_iters
        self.num_actions = 4
        super(GridworldModel, self).__init__(
            batch_size, feature_dim, gamma, query_size, discretization_const,
            true_reward_space_size, num_unknown, beta, beta_planner,
            objective, lr, discrete, optimize, args)

    def build_planner(self):
        height, width, dim = self.height, self.width, self.feature_dim
        num_actions, K = self.num_actions, self.K

        self.image = tf.compat.v1.placeholder(
            tf.float32, name="image", shape=[height, width])
        self.goals = tf.compat.v1.placeholder(
            tf.float32, name="goals", shape=[height, width])
        self.start = tf.compat.v1.placeholder(
            tf.float32, name="goals", shape=[height, width])
        self.grid = tf.stack([self.image, self.start, self.goals], axis=-1)
        self.features = tf.compat.v1.placeholder(
            tf.float32, name="features", shape=[height, width, dim])
        self.start_x = tf.compat.v1.placeholder(tf.int32, name="start_x", shape=[])
        self.start_y = tf.compat.v1.placeholder(tf.int32, name="start_y", shape=[])

        features_wall = tf.concat(
            [self.features, tf.expand_dims(self.image, -1)], axis=-1)
        features_wall = tf.stack([features_wall] * K, axis=0) # [K, height, width, dim]
        wall_constant = [[-1000000.0] for _ in range(K)]
        weights_wall = tf.concat([self.weights, wall_constant], axis=-1)
        # Change from K by dim to K by 1 by dim
        weights_wall = tf.expand_dims(weights_wall, axis=1)
        # Change to K by height by width by 1 by dim
        weights_wall = tf.stack([weights_wall] * width, axis=1)
        weights_wall = tf.stack([weights_wall] * height, axis=1) # [K, height, width, 1, dim]
        dim += 1

        feature_expectations = tf.zeros([K, height, width, dim])
        for i in range(self.num_iters):
            q_fes = self.bellman_update(feature_expectations, features_wall)
            q_values = tf.squeeze(tf.matmul(weights_wall, q_fes), [-2])
            if self.beta_planner == 'inf':
                best_actions = tf.argmax(q_values, axis=-1)
                self.policy = tf.one_hot(best_actions, 4) # previously: best_actions[0]
            else:
                self.policy = tf.nn.softmax(self.beta_planner * q_values, axis=-1)
            repeated_policy = tf.stack([self.policy] * dim, axis=-2)
            feature_expectations = tf.reduce_sum(
                tf.multiply(repeated_policy, q_fes), axis=-1)
            self.name_to_op['policy'+str(i)] = self.policy


        # Remove the wall feature
        self.feature_expectations_grid = feature_expectations[:,:,:,:-1]
        dim -= 1
        self.name_to_op['feature_exps_grid'] = self.feature_expectations_grid

        x, y = self.start_x, self.start_y
        self.feature_expectations = self.feature_expectations_grid[:,y,x,:]
        self.name_to_op['feature_exps'] = self.feature_expectations # has shape [K, dim]

        q_fes = self.bellman_update(feature_expectations, features_wall)
        q_values = tf.squeeze(tf.matmul(weights_wall, q_fes), [-2])
        self.q_values = q_values
        self.name_to_op['q_values'] = self.q_values
        self.name_to_op['policy'] = self.policy
        self.name_to_op['features'] = self.features
        self.name_to_op['grid'] = self.grid # [height, width, 3] -> (walls, start, goals)

        # self.run_build_planner = True

    def bellman_update(self, fes, features):
        height, width, dim = self.height, self.width, self.feature_dim + 1
        gamma, K = self.gamma, self.K
        extra_row = tf.zeros((K, 1, width, dim))
        extra_col = tf.zeros((K, height, 1, dim))

        north_lookahead = tf.concat([extra_row, fes[:,:-1]], axis=1)
        north_fes = features + gamma * north_lookahead
        south_lookahead = tf.concat([fes[:,1:], extra_row], axis=1)
        south_fes = features + gamma * south_lookahead
        east_lookahead = tf.concat([fes[:,:,1:], extra_col], axis=2)
        east_fes = features + gamma * east_lookahead
        west_lookahead = tf.concat([extra_col, fes[:,:,:-1]], axis=2)
        west_fes = features + gamma * west_lookahead
        return tf.stack([north_fes, south_fes, east_fes, west_fes], axis=-1)
    
    """-------------------------------------------RISK AVERSE PLANNER-------------------------------------"""
    
    def build_risk_planner(self):
        height, width, dim = self.height, self.width, self.feature_dim
        num_actions, N = self.num_actions, self.args.sample_size

        # get weight sample from distribution of reward functions
        # we need log_prior and true_reward_matrix
        # log prior is an array of log probabilities with shape (true_reward_space_size)
        # true_reward_matrix is the array of rewards with shape (true_reward_space_size, feature_dim)
        log_pr = tf.expand_dims(self.log_prior, axis=0) # add batch size
        weight_ids = tf.random.categorical(log_pr, num_samples = N) # get sample id's
        weight_ids = tf.squeeze(weight_ids, [0]) # remove batch_size
        weight_samples = tf.gather(self.true_reward_matrix, weight_ids) # get samples -> shape: [N, feature_dim]

        features_wall = tf.concat(
            [self.features, tf.expand_dims(self.image, -1)], axis=-1)
        # features_wall = tf.stack([features_wall] * N, axis=0)
        wall_constant = [[-1000000.0] for _ in range(N)]
        weights_wall = tf.concat([weight_samples, wall_constant], axis=-1) # still [N,dim]
        dim += 1

        feature_expectations = tf.zeros([height, width, dim])
        for i in range(self.num_iters):
            q_fes, q_values = self.risk_bellman_update(feature_expectations, features_wall, weights_wall) # it doesn't have the N dimension
            # q_fes shape: [height, width, dim, 4], q_values shape: [height, width, 4]
            # q_values = tf.squeeze(tf.matmul(weights_wall, q_fes), [-2])
            # here epsilon = 0, we have no exploration, only exploitation, so it is a greedy policy
            # we can improve that for more accurate results
            # (have the epsilon-greedy policy for learning and the greedy policy for acting)
            # actually that may not be needed because we check all possible actions to compute the q-values, exactly as the formula does
            if self.beta_planner == 'inf':
                best_actions = tf.argmax(q_values, axis=-1)
                self.risk_policy = tf.one_hot(best_actions, 4)
            else:
                self.risk_policy = tf.nn.softmax(self.beta_planner * q_values, axis=-1)
            # policy has shape: [height, width, 4]
            repeated_policy = tf.stack([self.risk_policy] * dim, axis=-2)
            # now [height, width, dim, 4]
            feature_expectations = tf.reduce_sum(
                tf.multiply(repeated_policy, q_fes), axis=-1)
            self.name_to_op['risk_policy'+str(i)] = self.risk_policy


        # Remove the wall feature
        self.risk_feature_expectations_grid = feature_expectations[:,:,:-1]
        dim -= 1
        self.name_to_op['risk_feature_exps_grid'] = self.risk_feature_expectations_grid

        x, y = self.start_x, self.start_y
        self.risk_feature_expectations = self.risk_feature_expectations_grid[y,x,:]
        self.name_to_op['risk_feature_exps'] = self.risk_feature_expectations
        # risk_feature_expectations has shape [dim] and weight_samples has shape [N, dim]
        # with these two we want to compute reward variance
        exp_reward = tf.tensordot(self.risk_feature_expectations, weight_samples, [[0],[1]]) # has shape [N]
        self.risk_variance = tf.math.reduce_std(exp_reward, axis=0)
        self.name_to_op['risk_variance'] = self.risk_variance

        q_fes, q_values = self.risk_bellman_update(feature_expectations, features_wall, weights_wall)
        self.risk_q_values = q_values
        self.name_to_op['risk_q_values'] = self.risk_q_values
        self.name_to_op['risk_policy'] = self.risk_policy
        # self.run_build_risk_planner = True

    def risk_reward(self, features, weights):
        # weights has dimensions [N, dim] and features [height, width, dim]
        exp_reward = tf.tensordot(features, weights, [[2],[1]]) # gives [height, width, N]
        var = tf.math.reduce_std(exp_reward, axis=2)
        mean = tf.math.reduce_mean(exp_reward, axis=2)
        min_r = tf.math.reduce_min(exp_reward, axis=2)
        living_r = tf.fill([self.height, self.width], self.args.living_reward)
        var_type = self.args.var_type
        var_coef = self.args.var_coef
        if(var_type == 'sub'):
            return mean - var_coef*var + living_r
        elif(var_type == 'div'):
            return tf.divide(mean+living_r, var)
        elif(var_type == 'worst'):
            return min_r + living_r
        elif(var_type == 'non_risk'):
            return mean + living_r
        elif(var_type == 'same'):
            return mean
    
    def risk_bellman_update(self, fes, features, weights_wall):
        height, width, dim = self.height, self.width, self.feature_dim + 1
        gamma, N = self.gamma, self.args.sample_size
        extra_row = tf.zeros((1, width, dim))
        extra_col = tf.zeros((height, 1, dim))

        # it has as observation (state) the whole grid features shifted by its position
        # will change it so that I first compute the reward for each state and then sum them
        # (so I will also have the weights_wall as an input and return q_values instead of q_fes)
        # get the rewards from self.args.var_type and self.args.var_coef
        # also, the learning rate is 1, it could be smaller for more accuracy
        north_lookahead = tf.concat([extra_row, fes[:-1]], axis=0)
        north_fes = features + gamma * north_lookahead
        south_lookahead = tf.concat([fes[1:], extra_row], axis=0)
        south_fes = features + gamma * south_lookahead
        east_lookahead = tf.concat([fes[:,1:], extra_col], axis=1)
        east_fes = features + gamma * east_lookahead
        west_lookahead = tf.concat([extra_col, fes[:,:-1]], axis=1)
        west_fes = features + gamma * west_lookahead
        q_fes = tf.stack([north_fes, south_fes, east_fes, west_fes], axis=-1)
        north_val = self.risk_reward(features, weights_wall) + gamma*self.risk_reward(north_lookahead, weights_wall)
        south_val = self.risk_reward(features, weights_wall) + gamma*self.risk_reward(south_lookahead, weights_wall)
        east_val = self.risk_reward(features, weights_wall) + gamma*self.risk_reward(east_lookahead, weights_wall)
        west_val = self.risk_reward(features, weights_wall) + gamma*self.risk_reward(west_lookahead, weights_wall)
        q_vals = tf.stack([north_val, south_val, east_val, west_val], axis=-1)
        return q_fes, q_vals

    """-------------------------------------RISK AVERSE Q LEARNING OPTIMIZED PLANNER------------------------------------"""

    def build_risk_q_planner(self):
        height, width, dim = self.height, self.width, self.feature_dim
        num_actions, N = self.num_actions, self.args.sample_size
        gamma, alpha = self.gamma, self.args.alpha
        # start position is [self.start_y, self.start_x]
        # end positions are the ones where we are on a wall or out of the grid
        # get weight sample from distribution of reward functions
        # we need log_prior and true_reward_matrix
        # log prior is an array of log probabilities with shape (true_reward_space_size)
        # true_reward_matrix is the array of rewards with shape (true_reward_space_size, feature_dim)
        log_pr = tf.expand_dims(self.log_prior, axis=0) # add batch size
        weight_ids = tf.random.categorical(log_pr, num_samples = N) # get sample id's
        weight_ids = tf.squeeze(weight_ids, [0]) # remove batch_size
        weight_samples = tf.gather(self.true_reward_matrix, weight_ids) # get samples -> shape: [N, feature_dim]

        features_wall = tf.concat(
            [self.features, tf.expand_dims(self.image, -1)], axis=-1) # add a new feature that is when a wall is in that position
        wall_constant = [[-1000000.0] for _ in range(N)]
        # adds the weight for the wall which is very negative (so the total reward is very negative as well)
        # so we don't need to care about walls at all
        weights_wall = tf.concat([weight_samples, wall_constant], axis=-1) # still [N,dim]
        dim += 1
        # we have the goals in one-hot array in self.goals
        q_values = tf.zeros([height, width, 4]) # table (state,action) -> q value
        for i in range(self.num_iters):
            # update q values with all possible actions in each case
            act_rewards = self.risk_q_actions(features_wall, weights_wall, self.goals)
            act_max_q = self.action_q_values(q_values)
            new_q_vals = q_values + alpha*(act_rewards + gamma*act_max_q - q_values)
            q_values = new_q_vals
        
        if self.beta_planner == 'inf':
            best_actions = tf.argmax(q_values, axis=-1)
            self.risk_policy = tf.one_hot(best_actions, 4) # dimensions [height, width, 4]
        else:
            self.risk_policy = tf.nn.softmax(self.beta_planner * q_values, axis=-1)
        
        # apply the policy for some iterations and create feature expectations
        dim-=1 # don't care about walls now
        self.risk_feature_expectations = tf.zeros([dim])
        state = tf.stack([self.start_y, self.start_x])
        features = self.features
        risk_policy = tf.concat([self.risk_policy, tf.zeros([height, width, 1])], axis=-1) # add a 5th move which is 'stay where you are'
        discount = 1.0
        for i in range(self.num_iters):
            # we need to discount the rewards (and the features)
            self.risk_feature_expectations += discount*features[state[0]][state[1]]
            act = tf.argmax(risk_policy[state[0]][state[1]])
            # now move features and policy arrays as before (because we cannot check if the state is out of bounds)
            features, risk_policy = self.res_act(features, risk_policy, act)
            discount = discount*gamma
        
        self.risk_q_values = q_values
        self.name_to_op['risk_q_values'] = self.risk_q_values
        self.name_to_op['risk_policy'] = self.risk_policy
        self.name_to_op['risk_feature_exps'] = self.risk_feature_expectations
        # risk_feature_expectations has shape [dim] and weight_samples has shape [N, dim]
        # with these two we want to compute reward variance
        exp_reward = tf.tensordot(self.risk_feature_expectations, weight_samples, [[0],[1]]) # has shape [N]
        self.risk_variance = tf.math.reduce_std(exp_reward, axis=0)
        self.name_to_op['risk_variance'] = self.risk_variance
    
    def res_act(self, features, policy, act):
        height, width, dim = self.height, self.width, self.feature_dim
        extra_row = tf.zeros((1, width, dim))
        extra_col = tf.zeros((height, 1, dim))

        north_features = tf.concat([extra_row, features[:-1]], axis=0)
        south_features = tf.concat([features[1:], extra_row], axis=0)
        east_features = tf.concat([features[:,1:], extra_col], axis=1)
        west_features = tf.concat([extra_col, features[:,:-1]], axis=1)
        act_features = tf.stack([north_features, south_features, east_features, west_features, features], axis=-2)

        extra_policy_row = tf.zeros((1, width, 4))
        extra_policy_col = tf.zeros((height, 1, 4))
        # add 1 to the 5th move
        extra_policy_row = tf.concat([extra_policy_row, tf.fill([1, width, 1],1.0)], axis=-1)
        extra_policy_col = tf.concat([extra_policy_col, tf.fill([height, 1, 1],1.0)], axis=-1)

        north_policy = tf.concat([extra_policy_row, policy[:-1]], axis=0)
        south_policy = tf.concat([policy[1:], extra_policy_row], axis=0)
        east_policy = tf.concat([policy[:,1:], extra_policy_col], axis=1)
        west_policy = tf.concat([extra_policy_col, policy[:,:-1]], axis=1)
        act_policy = tf.stack([north_policy, south_policy, east_policy, west_policy, policy], axis=-2)

        return act_features[:,:,act], act_policy[:,:,act]
    
    def risk_q_actions(self, features, weight_wall, goals):
        height, width, dim = self.height, self.width, self.feature_dim + 1
        extra_row = tf.zeros((1, width, dim))
        extra_col = tf.zeros((height, 1, dim))

        north_features = tf.concat([extra_row, features[:-1]], axis=0)
        north_reward = self.risk_reward(north_features, weight_wall)
        south_features = tf.concat([features[1:], extra_row], axis=0)
        south_reward = self.risk_reward(south_features, weight_wall)
        east_features = tf.concat([features[:,1:], extra_col], axis=1)
        east_reward = self.risk_reward(east_features, weight_wall)
        west_features = tf.concat([extra_col, features[:,:-1]], axis=1)
        west_reward = self.risk_reward(west_features, weight_wall)
        act_rewards = tf.stack([north_reward, south_reward, east_reward, west_reward], axis=-1) # shape: [height, width, 4]
        # now make the rewards in goal states 0 (it is one-hot)
        all_1 = tf.fill([height,width], 1.0)
        invert_goals = all_1-goals # now we only have 0 when in a goal
        invert_goals = tf.stack([invert_goals]*4, axis=-1) # has shape [height, width, 4]
        act_rewards = tf.multiply(act_rewards, invert_goals) # has the same shape, and all 0 in 4 directions when on a goal
        return act_rewards

    def action_q_values(self, q_values):
        height, width= self.height, self.width
        extra_row = tf.zeros((1, width, 4)) # q values placeholder for out of grid
        extra_col = tf.zeros((height, 1, 4))
        north_q_values = tf.concat([extra_row, q_values[:-1]], axis=0)
        north_max_q = tf.reduce_max(north_q_values, axis=-1)
        south_q_values = tf.concat([q_values[1:], extra_row], axis=0)
        south_max_q = tf.reduce_max(south_q_values, axis=-1)
        east_q_values = tf.concat([q_values[:,1:], extra_col], axis=1)
        east_max_q = tf.reduce_max(east_q_values, axis=-1)
        west_q_values = tf.concat([extra_col, q_values[:,:-1]], axis=1)
        west_max_q = tf.reduce_max(west_q_values, axis=-1)
        act_max_q = tf.stack([north_max_q, south_max_q, east_max_q, west_max_q], axis=-1) # shape: [height, width, 4]
        return act_max_q
        

    """--------------------------------NON RISK AVERSE Q LEARNING PLANNER------------------------------------------------"""
    
    def build_non_risk_q_planner(self):
        height, width, dim = self.height, self.width, self.feature_dim
        K = self.K
        gamma, alpha = self.gamma, self.args.alpha
        
        features_wall = tf.concat(
            [self.features, tf.expand_dims(self.image, -1)], axis=-1) # [height, width, dim]
        wall_constant = [[-1000000.0] for _ in range(K)]
        weights_wall = tf.concat([self.weights, wall_constant], axis=-1) # [K, dim]
        dim += 1

        # we have the goals in one-hot array in self.goals
        q_values = tf.zeros([K, height, width, 4]) # table (state,action) -> q value
        for i in range(self.num_iters):
            # update q values with all possible actions in each case
            act_rewards = self.non_risk_q_actions(features_wall, weights_wall, self.goals) # [K, height, width, 4]
            act_max_q = self.non_risk_action_q_values(q_values) # same shape
            new_q_vals = q_values + alpha*(act_rewards + gamma*act_max_q - q_values)
            q_values = new_q_vals
        
        if self.beta_planner == 'inf':
            best_actions = tf.argmax(q_values, axis=-1)
            self.non_risk_policy = tf.one_hot(best_actions, 4) # dimensions [K, height, width, 4]
        else:
            self.non_risk_policy = tf.nn.softmax(self.beta_planner * q_values, axis=-1)
        
        # apply the policy for some iterations and create feature expectations
        dim-=1 # don't care about walls now
        self.non_risk_feature_expectations = tf.zeros([K, dim])
        state = tf.stack([self.start_y, self.start_x])
        features = tf.stack([self.features]*K, axis=0)
        non_risk_policy = tf.concat([self.non_risk_policy, tf.zeros([K, height, width, 1])], axis=-1) # add a 5th move which is 'stay where you are'
        discount = 1.0
        for i in range(self.num_iters):
            # we need to discount the rewards (and the features)
            self.non_risk_feature_expectations += discount*features[:, state[0], state[1]]
            # now move features and policy arrays as before (because we cannot check if the state is out of bounds)
            features, non_risk_policy = self.non_risk_res_act(features, non_risk_policy, state)
            discount = discount*gamma
        
        self.non_risk_q_values = q_values
        self.name_to_op['non_risk_q_values'] = self.non_risk_q_values
        self.name_to_op['non_risk_policy'] = self.non_risk_policy
        self.name_to_op['non_risk_feature_exps'] = self.non_risk_feature_expectations
    
    def non_risk_res_act(self, features, policy, state):
        height, width, dim = self.height, self.width, self.feature_dim
        extra_row = tf.zeros((self.K, 1, width, dim))
        extra_col = tf.zeros((self.K, height, 1, dim))

        north_features = tf.concat([extra_row, features[:,:-1]], axis=1)
        south_features = tf.concat([features[:,1:], extra_row], axis=1)
        east_features = tf.concat([features[:,:,1:], extra_col], axis=2)
        west_features = tf.concat([extra_col, features[:,:,:-1]], axis=2)
        act_features = tf.stack([north_features, south_features, east_features, west_features, features], axis=1) # [K, 5, height, width, dim]

        extra_policy_row = tf.zeros((self.K, 1, width, 4))
        extra_policy_col = tf.zeros((self.K, height, 1, 4))
        # add 1 to the 5th move
        extra_policy_row = tf.concat([extra_policy_row, tf.fill([self.K, 1, width, 1],1.0)], axis=-1)
        extra_policy_col = tf.concat([extra_policy_col, tf.fill([self.K, height, 1, 1],1.0)], axis=-1)

        north_policy = tf.concat([extra_policy_row, policy[:,:-1]], axis=1)
        south_policy = tf.concat([policy[:,1:], extra_policy_row], axis=1)
        east_policy = tf.concat([policy[:,:,1:], extra_policy_col], axis=2)
        west_policy = tf.concat([extra_policy_col, policy[:,:,:-1]], axis=2)
        act_policy = tf.stack([north_policy, south_policy, east_policy, west_policy, policy], axis=1) # [K, 5, height, width, 5] (the first '5' is the action we will do now)
        act = tf.argmax(policy[:, state[0], state[1], :], axis=-1) # [K]
        one_hot_act = tf.one_hot(act, 5) # [K, 5] one-hot
        one_hot_act = tf.stack([one_hot_act]*height, axis=-1)
        one_hot_act = tf.stack([one_hot_act]*width, axis=-1)
        one_hot_act_features = tf.stack([one_hot_act]*dim, axis=-1) # [K, 5, height, width, dim]
        one_hot_act_policy = tf.stack([one_hot_act]*5, axis=-1) # [K, 5 , height, width, 5]
        act_features = tf.multiply(act_features, one_hot_act_features)
        act_features = tf.reduce_sum(act_features, axis=1)
        act_policy = tf.multiply(act_policy, one_hot_act_policy)
        act_policy = tf.reduce_sum(act_policy, axis=1)

        return act_features, act_policy

    
    def non_risk_q_actions(self, features, weight_wall, goals):
        height, width, dim = self.height, self.width, self.feature_dim + 1
        extra_row = tf.zeros((1, width, dim))
        extra_col = tf.zeros((height, 1, dim))

        north_features = tf.concat([extra_row, features[:-1]], axis=0)
        north_reward = self.non_risk_reward(north_features, weight_wall)
        south_features = tf.concat([features[1:], extra_row], axis=0)
        south_reward = self.non_risk_reward(south_features, weight_wall)
        east_features = tf.concat([features[:,1:], extra_col], axis=1)
        east_reward = self.non_risk_reward(east_features, weight_wall)
        west_features = tf.concat([extra_col, features[:,:-1]], axis=1)
        west_reward = self.non_risk_reward(west_features, weight_wall)
        act_rewards = tf.stack([north_reward, south_reward, east_reward, west_reward], axis=-1) # shape: [K, height, width, 4]
        # now make the rewards in goal states 0 (it is one-hot)
        # all_1 = tf.fill([height,width], 1.0)
        # invert_goals = all_1-goals # now we only have 0 when in a goal
        # invert_goals = tf.stack([invert_goals]*4, axis=-1) # has shape [height, width, 4]
        # invert_goals = tf.stack([invert_goals]*self.K, axis=0) # has shape [K, height, width, 4]
        # act_rewards = tf.multiply(act_rewards, invert_goals) # has the same shape, and all 0 in 4 directions when on a goal
        return act_rewards

    def non_risk_action_q_values(self, q_values):
        height, width= self.height, self.width
        extra_row = tf.zeros((self.K, 1, width, 4)) # q values placeholder for out of grid
        extra_col = tf.zeros((self.K, height, 1, 4))
        north_q_values = tf.concat([extra_row, q_values[:,:-1]], axis=1)
        north_max_q = tf.reduce_max(north_q_values, axis=-1)
        south_q_values = tf.concat([q_values[:,1:], extra_row], axis=1)
        south_max_q = tf.reduce_max(south_q_values, axis=-1)
        east_q_values = tf.concat([q_values[:,:,1:], extra_col], axis=2)
        east_max_q = tf.reduce_max(east_q_values, axis=-1)
        west_q_values = tf.concat([extra_col, q_values[:,:,:-1]], axis=2)
        west_max_q = tf.reduce_max(west_q_values, axis=-1)
        act_max_q = tf.stack([north_max_q, south_max_q, east_max_q, west_max_q], axis=-1) # shape: [K, height, width, 4]
        return act_max_q
    
    def non_risk_reward(self, features, weights):
        # weights has dimensions [K, dim] and features [height, width, dim]
        non_living_reward = tf.tensordot(weights, features, [[1],[2]]) # [K, height, width]
        living_r = tf.fill([self.K, self.height, self.width], self.args.living_reward)
        # rew = non_living_reward + living_r
        rew = non_living_reward
        return rew
        
    
    def update_feed_dict_with_mdp(self, mdp, fd):
        image, features, start_state = mdp.convert_to_numpy_input()
        x, y = start_state
        fd[self.image] = image # this contains the walls in bool
        fd[self.features] = features
        fd[self.start_x] = x
        fd[self.start_y] = y
        # fd[self.goals] = mdp.goals
        goals = np.zeros([self.height, self.width], dtype=np.int32)
        for i,j,_ in mdp.goals:
            goals[j][i]=1
        fd[self.goals] = goals
        start = np.zeros([self.height, self.width], dtype=np.int32)
        start[y][x]=1
        fd[self.start] = start


class NoPlanningModel(Model):

    def build_weights(self):
        pass

    def build_planner(self):
        self.feature_expectations = tf.compat.v1.placeholder(
            tf.float32, shape=[self.K, self.feature_dim], name='feature_exps')
        self.name_to_op['feature_exps'] = self.feature_expectations
    
    def build_risk_planner(self):
        pass

    def update_feed_dict_with_mdp(self, mdp, fd):
        pass
