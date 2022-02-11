import time
from collections import deque

import numpy as np
import torch

import torch
import torch.nn as nn
import torch.nn.functional as F

"""
In this file are created the classes related to the RL machinery (environment, agent, ...)
"""

np.set_printoptions(precision=2)
np.set_printoptions(suppress=True)
torch.set_printoptions(sci_mode=False, precision=3)

class Agent(object):
    """ Agent abstract class """

    def __init__(self):
        pass

    def act(self, context):
        raise NotImplementedError()

    def train(self):
        pass

    def eval(self):
        pass


class AgentDN(Agent):
    """ Deep Net Agent abstract class """

    def __init__(self, nets):
        super(AgentDN, self).__init__()
        self.nets = nets

    def evaluate(self, context, importance_weights):
        predictions, acc_weights = 0, 0.
        for idx, net in enumerate(self.nets):
            if len(context.shape) > 2:
                predictions += net(context).mean(dim=1) * (importance_weights[idx] + 1e-10)
            else:
                predictions += net(context).squeeze() * (importance_weights[idx] + 1e-10)
            acc_weights += (importance_weights[idx] + 1e-10)
        return predictions / acc_weights

    def train(self):
        for net in self.nets:
            net.train()

    def eval(self):
        for net in self.nets:
            net.eval()


class AgentGreedy(AgentDN):
    """ Greedy Agent: chooses best action with probability (1 - epsilon), random one with probability epsilon """

    def __init__(self, nets, epsilon):
        super(AgentGreedy, self).__init__(nets)
        self.epsilon = epsilon

    def act(self, context, importance_weights):
        predictions = self.evaluate(context, importance_weights)
        filtr = (torch.rand(context.shape[0]) > self.epsilon).long()
        action = predictions.argmax(dim=-1).squeeze() * filtr + torch.randint(0, 2, (context.shape[0],)) * (1 - filtr)
        return action


class AgentBayesNet(AgentDN):
    def __init__(self, bayes_net, sample):
        """ Initialize Bayesian net Agent
        :param sample: int
            number of samples to estimate rewards
        """
        super(AgentBayesNet, self).__init__(bayes_net)
        self.sample = sample

    def act(self, context, importance_weights=[1.] * 1000):
        if len(context.shape) < 3:
            context = context.unsqueeze(0)
        predictions = self.evaluate(torch.repeat_interleave(context, self.sample, dim=1), importance_weights)
        return predictions.argmax(dim=-1).squeeze()


class AgentDropout(AgentBayesNet):
    """ Dropout Agent (seen as Bayesian one) """

    def __init__(self, net, sample=2):
        super(AgentDropout, self).__init__(net, sample)


class AgentBayesBackprop(AgentBayesNet):
    """ Bayes by Backprop Agent """

    def __init__(self, net, sample=2):
        super(AgentBayesBackprop, self).__init__(net, sample)


class EnvMushroom(object):
    """ Environment based on mushroom dataset
    Mushrooms have a list of features and are either edible (0) or poisonous (1)
    Remark: environment state is only charactrized by an integer referring to a row in the mushroom dataset
    """

    def __init__(self, context_inds, classes):
        self.context_inds = context_inds
        self.classes = classes
        self.context_ind = None

    @property
    def is_edible(self):
        return self.classes[self.context_ind] == 0

    @property
    def oracle(self):
        return 5 if self.is_edible else 0

    def reset(self):
        self.context_ind = np.random.randint(0, len(self.context_inds))
        return self.context_ind

    def step(self, action):
        assert action in [0, 1]
        if action == 0:
            reward = 0  # do not eat -> reward is 0
        elif self.is_edible:
            # eat an edible mushroom
            reward = 5
        else:
            # eat a poisonous mushroom
            reward = np.random.choice([-35, 5])
        return self.reset(), reward


class ReplayBuffer(object):
    """ Replay buffer from which minibatches of [contexts, actions, rewards] are drawn for training """

    def __init__(self, max_len):
        self.buffer = deque(maxlen=max_len)

    def sample(self, batch_size):
        assert batch_size < len(self)
        indices = np.random.choice(len(self), batch_size, replace=False)  # select indices of selected elements
        context_inds, actions, rewards = zip(*[self.buffer[idx] for idx in indices])
        return context_inds, actions, rewards

    def add(self, context_ind, action, reward):
        self.buffer.append([context_ind, action, reward])

    def __len__(self):
        return len(self.buffer)



class RLReg(object):
    """ Abstract class for training an AgentDN """

    def __init__(self, X_train, y_train, agent, buffer_size=4096, minibatch_size=64, burn_in=256):
        self.agent = agent
        self.buffer = ReplayBuffer(buffer_size)
        self.X_train = X_train
        self.env = EnvMushroom(np.arange(len(X_train)), y_train)
        self.episode = 0
        self.regret = 0
        self.burn_in = burn_in
        self.minibatch_size = minibatch_size
        self.context_ind = self.env.reset()
        self.hist = {'regret': [self.regret]}

    def train(self, episodes, pull, iter_trains, total_data, stepsize, warm_up, decay, samplers, CUDA_EXISTS):
        self.agent.train()
        importance_weights = [1.] * len(samplers)
        for episode_ in range(episodes):
            for _ in range(pull):
                self.episode += 1
                oracle = self.env.oracle
                if self.episode < self.burn_in:  # random action
                    action = np.random.randint(0, 2)
                else:
                    action = self.agent.act(self.X_train[self.context_ind], importance_weights).item()
                next_context_ind, reward = self.env.step(action)

                self.buffer.add(self.context_ind, action, reward)
                self.context_ind = next_context_ind
                self.regret += oracle - reward
            self.hist['regret'].append(self.regret)

            if len(self.buffer) > self.burn_in:
                for idx, sampler in enumerate(samplers):
                    sampler.lr *= decay
                for _ in range(iter_trains):
                    losses = []
                    for idx, sampler in enumerate(samplers):
                        sampler.zero_grad()
                        context_inds, actions, rewards = self.buffer.sample(self.minibatch_size)
                        context_inds, actions, rewards = np.array(context_inds), np.array(actions), torch.tensor(
                            np.array(rewards, dtype=float)).float()
                        if CUDA_EXISTS:
                            rewards = rewards.cuda()
                        loss = self.get_loss_(sampler.net, context_inds, actions, rewards) * total_data
                        loss.backward(retain_graph=True)
                        importance_weights[idx] = sampler.step(loss.item())
                        losses.append(loss.item())
                    if sampler.c == 'csgld' and episode_ > warm_up * episodes:
                        randomField, cnt_weights = 0, 1e-10
                        for idx in range(len(samplers)):
                            if samplers[idx].J < samplers[idx].part - 1:
                                randomField += samplers[idx].randomField
                                cnt_weights += 1
                        for idx, _ in enumerate(samplers):
                            samplers[idx].update_H(randomField / cnt_weights, stepsize)
                
                if episode_ % 20 == 0:
                    idx = 0
                    print('epoch: {} / {} | loss: {:9.1f} | regret: {} | index: {} | lr: {:3.2e} multiplier: {:3.3f} min: {:3.3f} max: {:.3f} weight: {:.3f}'.format(\
                            episode_, episodes, losses[idx], self.regret, samplers[idx].J, sampler.lr, samplers[idx].gmul, \
                            samplers[idx].gmul_min, samplers[idx].gmul_max, importance_weights[idx]))
        return

    def get_loss_(self, context_inds, actions, rewards):
        """ Function called in the main training loop to get the loss to minimize """
        raise NotImplementedError()


class DeterministicRLNet(nn.Module):
    """ Defines a neural network with two hidden layers of size hidden_size. A
        relu activation is applied after the hidden layer.
    """
    def __init__(self, dim_context, hidden_size, dim_action_space):
        super(DeterministicRLNet, self).__init__()
        self.fc1 = nn.Linear(dim_context, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, dim_action_space)

    def forward(self, x):
        out = F.relu(self.fc1(x))
        out = F.relu(self.fc2(out))
        return self.fc3(out)

class DeterministicRLReg(RLReg):
    """ Class for training an AgentDN """
    def __init__(self, X_train, y_train, agent, criterion=torch.nn.MSELoss(), buffer_size=4096, minibatch_size=64,
                 burn_in=500):
        super(DeterministicRLReg, self).__init__(X_train, y_train, agent, buffer_size, minibatch_size, burn_in)
        self.criterion = criterion

    def get_loss_(self, net, context_inds, actions, rewards):
        context = self.X_train[context_inds]
        if len(context.shape) > 2:
            rewards_preds = net(context).mean(dim=1)
        else:
            rewards_preds = net(context).squeeze()
        reward_preds = rewards_preds[np.arange(self.minibatch_size), actions]
        loss = self.criterion(reward_preds, rewards)
        return loss


class DropoutNet(nn.Module):
    """ Defines a neural network with one hidden layer with size hidden_size and
        a relu activation.
        Applies dropout with probability p after the relu function.
    """

    def __init__(self, dim_input, hidden_size, dim_output, p):
        super(DropoutNet, self).__init__()
        self.fc1 = nn.Linear(dim_input, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, dim_output)
        self.p = p

    def forward(self, x):
        out = F.relu(self.fc1(x))
        out = F.dropout(F.relu(self.fc2(out)), self.p)
        return self.fc3(out)

class DropoutRLNet(DropoutNet):
    """ Essentially similar to a DropoutNet defined in dropout_regression.py """

    def __init__(self, hidden_size, dim_context, dim_action_space, p):
        super(DropoutRLNet, self).__init__(hidden_size, dim_input=dim_context, dim_output=dim_action_space, p=p)


class DropoutRLReg(RLReg):
    """ Class for training an AgentB """

    def __init__(self, X_train, y_train, agent, buffer_size=4096, minibatch_size=64, burn_in=500,
                 criterion=torch.nn.MSELoss()):
        super(DropoutRLReg, self).__init__(X_train, y_train, agent, buffer_size, minibatch_size, burn_in)
        self.criterion = criterion

    def get_loss_(self, net, context_inds, actions, rewards):
        context = self.X_train[context_inds]
        if len(context.shape) < 3:
            context = context.unsqueeze(0)

        if len(context.shape) > 2:
            rewards_preds = net(torch.repeat_interleave(context, self.agent.sample, dim=1)).mean(dim=1)
        else:
            rewards_preds = net(torch.repeat_interleave(context, self.agent.sample, dim=1)).squeeze()

        reward_preds = rewards_preds[np.arange(self.minibatch_size), actions]
        loss = self.criterion(reward_preds, rewards)
        return loss



""" BayesBackProp """


class VarPosterior(object):
    """ Defines the variational posterior distribution q(w) for the weights of
        the network.

        Here we suppose that q(w) = N(mu , log(1 + exp(rho))).
    """

    def __init__(self, mu, rho):
        self.mu = mu
        self.rho = rho
        self.gaussian = torch.distributions.Normal(0, 1)

    @property
    def sigma(self):
        """Returns the variance of the distribution sigma = log(1 + exp(rho))"""
        return torch.log1p(torch.exp(self.rho))

    def sample(self):
        """Samples mu + sigma*N(0 , 1)"""
        epsilon = self.gaussian.sample(self.rho.size())
        return self.mu + self.sigma * epsilon

    def log_prob(self, x):
        """Returns the log-distribution of a vector x whose each component is
           independently distributed according to q(w).
        """
        return torch.sum(-0.5 * torch.log(2 * np.pi * self.sigma ** 2)
                         - 0.5 * (x - self.mu) ** 2 / self.sigma ** 2)


class Prior(object):
    """ Defines the prior distribution p(w) for the weights of the network.

        Here we suppose that p(w) = pi*N(0 , sigma1) + (1 - pi)*N(0 , sigma2).
    """

    def __init__(self, sigma1, sigma2, pi):
        self.sigma1 = sigma1
        self.sigma2 = sigma2
        self.pi = pi
        self.gaussian1 = torch.distributions.Normal(0, sigma1)
        self.gaussian2 = torch.distributions.Normal(0, sigma2)

    def sample(self):
        """ Samples x*N(0 , sigma1) + (1 - x)*N(0 ,1), where x follows a Bernoulli
            laws of parameter pi.
        """
        x = np.random.binomial(1, self.pi)
        return x * self.gaussian1.sample(torch.Size([1])) + (1 - x) * self.gaussian2.sample(torch.Size([1]))

    def log_prob(self, x):
        """Returns the log-distribution of a vector x whose each component is
           independently distributed according to p(w).

           To deal with overflows we compute:
               log(pi) + log(Gauss1) + log(1 + (1 - pi)/pi * Gauss2/Gauss1)
        """
        function = lambda x: x * np.exp(-x ** 2)
        return torch.sum(np.log(self.pi) + self.gaussian1.log_prob(x)
                         + np.log1p(((1 - self.pi) / self.pi) * function(self.sigma1 / self.sigma2)))


class BayesianLinear(nn.Module):
    """ Defines a linear layer for a neural network.

        The weights w are distributed according to the posterior q(w ; w_mu , w_rho)
        The biases b are also distributed according to the posterior q(b ; b_mu , b_rho)

        w and b are associated with priors p(w ; sigma1,sigma2,pi) and p(b ; sigma1,sigma2,pi)
    """

    def __init__(self, dim_input, dim_output, prior_parameters):
        super(BayesianLinear, self).__init__()

        self.dim_input = dim_input
        self.dim_output = dim_output
        self.prior_parameters = prior_parameters
        
        """ follow Bayesian deep bandit shutdown to use a smaller std 0.05 to encourage MAP first instead of 1 """
        new_var = 0.05
        self.w_mu = nn.Parameter(torch.Tensor(dim_output, dim_input).normal_(0, new_var))
        self.w_rho = nn.Parameter(torch.Tensor(dim_output, dim_input).normal_(0, new_var))
        self.w = VarPosterior(self.w_mu, self.w_rho)
        self.w_prior = Prior(prior_parameters['sigma1'], prior_parameters['sigma2'], prior_parameters['pi'])

        self.b_mu = nn.Parameter(torch.Tensor(dim_output).normal_(0, 1))
        self.b_rho = nn.Parameter(torch.Tensor(dim_output).normal_(0, 1))
        self.b = VarPosterior(self.b_mu, self.b_rho)
        self.b_prior = Prior(prior_parameters['sigma1'], prior_parameters['sigma2'], prior_parameters['pi'])

        self.log_prior = 0
        self.log_variational_posterior = 0

    def forward(self, x):
        """ Samples a couple (w , b) with the variational posteriors, saves the
            log-likelihoods of this sample and returns the output of the layer
            computed with these samples.
        """
        w = self.w.sample()
        b = self.b.sample()

        self.log_prior = self.w_prior.log_prob(w) + self.b_prior.log_prob(b)
        self.log_variational_posterior = self.w.log_prob(w) + self.b.log_prob(b)

        return F.linear(x, w, b)

    def get_weights_mu(self):
        """ Auxiliary function used to get the weight distribution of a net """
        return np.hstack([self.w_mu.detach().numpy().flatten(), self.b_mu.detach().numpy().flatten()])


class BayesBackpropRLNet(nn.Module):
    """ Defines a neural-network with one hidden layer with size hidden-size and
        relu activation. Each layer is a BayesianLinear layer defined as above.
        Builds the ELBO function associated with the network:
            KL(q(w) ||p(w)) - E[log(p(D | w))]  (w: all the parameters of the network)
    """

    def __init__(self, dim_context, hidden_size, dim_action_space, prior_parameters, sigma):
        super(BayesBackpropRLNet, self).__init__()
        self.fc1 = BayesianLinear(dim_input=dim_context, dim_output=hidden_size
                                  , prior_parameters=prior_parameters)
        self.fc2 = BayesianLinear(dim_input=hidden_size, dim_output=hidden_size
                                  , prior_parameters=prior_parameters)
        self.fc3 = BayesianLinear(dim_input=hidden_size, dim_output=dim_action_space
                                  , prior_parameters=prior_parameters)
        self.sigma = sigma  # noise associated with the data y = f(x; w) + N(0, self.sigma)

        self.layers = [self.fc1, self.fc2, self.fc3]

    def forward(self, x):
        out = F.relu(self.fc1(x))
        out = F.relu(self.fc2(out))
        out = self.fc3(out)
        return out

    def log_prior(self):
        """ Computes log(p(w)) """
        return sum(map(lambda fc: fc.log_prior, self.layers))

    def log_variational_posterior(self):
        """ Computes log(q(w|D)) """
        return sum(map(lambda fc: fc.log_variational_posterior, self.layers))

    def log_likelihood(self, y, output):
        """ Computes log(p(D|w))
            Rmk: y_i = f(x_i ; w) + epsilon (epsilon ~ N(0 , self.sigma))
                 So we have p(y_i | x_i , w) = N(f(x_i ; w) , self.sigma)
        """
        return torch.sum(-0.5 * np.log(2 * np.pi * self.sigma ** 2) - 0.5 * (y - output) ** 2 / self.sigma ** 2)

    def sample_elbo(self, x, y, actions, MC_samples):
        """ For a batch x computes weight * E(log(q(w)) - log(p(w))) - E(log(p(D |w)))
            The expected values are computed with a MC scheme (at each step w is sampled
            from q(w)).
        """
        log_var_posterior = self.log_variational_posterior()
        log_prior = self.log_prior()

        log_likelihoods = 0
        outs = self.forward(torch.repeat_interleave(x, MC_samples, dim=1))[np.arange(x.shape[0]), ..., actions]
        for i in range(MC_samples):
            log_likelihood = self.log_likelihood(y, outs[:, i])
            log_likelihoods += log_likelihood
        log_likelihoods /= MC_samples

        elbo = log_var_posterior - log_prior - log_likelihoods

        return elbo, log_var_posterior, log_prior, log_likelihoods


class BayesRLReg(RLReg):
    """ Class for training an AgentB """

    def __init__(self, X_train, y_train, agent, buffer_size=4096, minibatch_size=64, burn_in=500):
        super(BayesRLReg, self).__init__(X_train, y_train, agent, buffer_size, minibatch_size, burn_in)

    def get_loss_(self, net, context_inds, actions, rewards):
        loss, log_var_posterior, log_prior, log_likelihood = net.sample_elbo(self.X_train[context_inds],
                                                                                        rewards, actions,
                                                                                        self.agent.sample,
                                                                                        )
        return loss


