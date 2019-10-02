import matplotlib.pylab as plt
import numpy as np
from scipy.stats import norm
import seaborn as sns
import torch
from hmc import *
import ipdb

def get_schedule(num, rad=4):
    if num == 1:
        return np.array([0.0, 1.0])
    t = np.linspace(-rad, rad, num)
    s = 1.0 / (1.0 + np.exp(-t))
    return ((s - np.min(s)) / (np.max(s) - np.min(s))).astype('float32')

def log_mean_exp(x, axis=None):
    assert axis is not None
    m = torch.max(x, dim=axis, keepdim=True)[0]
    return m + torch.log(torch.mean(
        torch.exp(x - m), dim=axis, keepdim=True))


def flatten_sum(logps):
    while len(logps.size()) > 1:
        logps = logps.sum(dim=-1)
    return logps

def standard_gaussian(shape):
    mean, logsd = [torch.FloatTensor(*shape).fill_(0.) for _ in range(2)]
    return gaussian_diag(mean, logsd)


def gaussian_diag(mean, logsd):
    class o(object):
        Log2PI = float(np.log(2 * np.pi))
        pass

        def logps(x):
            return  -0.5 * (o.Log2PI + 2. * logsd + ((x - mean) ** 2) / torch.exp(2. * logsd))

        def sample():
            eps = torch.zeros_like(mean).normal_()
            return mean + torch.exp(logsd) * eps

    o.logp = lambda x: flatten_sum(o.logps(x))
    return o



class AIS(object):
    def __init__(self, x_ph, log_likelihood_fn, dims, num_samples,
                 stepsize=0.01, n_steps=10,
                 target_acceptance_rate=.65, avg_acceptance_slowness=0.9,
                 stepsize_min=0.0001, stepsize_max=0.5, stepsize_dec=0.98, stepsize_inc=1.02):
        """
        The model implements Hamiltonian AIS.
        Developed by @bilginhalil on top of https://github.com/jiamings/ais/

        Example use case:
        logp(x|z) = |integrate over z|{logp(x|z,theta) + logp(z)}
        p(x|z, theta) -> likelihood function p(z) -> prior
        Prior is assumed to be a normal distribution with mean 0 and identity covariance matrix

        :param x_ph: Placeholder for x
        :param log_likelihood_fn: Outputs the logp(x|z, theta), it should take two parameters: x and z
        :param e.g. {'output_dim': 28*28, 'input_dim': FLAGS.d, 'batch_size': 1} :)
        :param num_samples: Number of samples to sample from in order to estimate the likelihood.

        The following are parameters for HMC.
        :param stepsize:
        :param n_steps:
        :param target_acceptance_rate:
        :param avg_acceptance_slowness:
        :param stepsize_min:
        :param stepsize_max:
        :param stepsize_dec:
        :param stepsize_inc:
        """

        self.dims = dims
        self.log_likelihood_fn = log_likelihood_fn
        self.num_samples = num_samples

        self.z_shape = [dims['batch_size'] * self.num_samples, dims['input_dim']]
#         print(self.z_shape)
        self.prior = standard_gaussian(self.z_shape)

        self.batch_size = dims['batch_size']
        self.x = x_ph

        self.stepsize = stepsize
        self.avg_acceptance_rate = target_acceptance_rate
        self.n_steps = n_steps
        self.stepsize_min = stepsize_min
        self.stepsize_max = stepsize_max
        self.stepsize_dec = stepsize_dec
        self.stepsize_inc = stepsize_inc
        self.target_acceptance_rate = target_acceptance_rate
        self.avg_acceptance_slowness = avg_acceptance_slowness

    def log_f_i(self, z, t):

        return (- self.energy_fn(z, t)).view(self.num_samples, self.batch_size)

    def energy_fn(self, z, t):
#         print(self.x.shape)
        e = self.prior.logp(z) + t * \
            self.log_likelihood_fn(self.x, z).view(
                       self.num_samples * self.batch_size)

        return -e

    def ais(self, z, schedule):
        """
            :param z: initial samples drawn from prior, with shape [num_samples*batch_size]
            :param schedule: temperature schedule i.e. `p(z)p(x|z)^t`
        """

        

        items = torch.from_numpy(np.array([[i, t0, t1] for i, (t0, t1) in 
                                      enumerate(zip(schedule[:-1], schedule[1:]))]))

        def condition(index, summation, z, stepsize, avg_acceptance_rate):
            return index < len(schedule)-1

        def body(index, w, z, stepsize, avg_acceptance_rate):
            item = items[index]
            t0 = item[1]
            t1 = item[2]

            new_u = self.log_f_i(z, t1)
            prev_u = self.log_f_i(z, t0)
            w = w + (new_u - prev_u)

            def run_energy(z):
                return self.energy_fn(z, t1)
                

            # New step:
            accept, final_pos, final_vel = hmc_move(
                z,
                run_energy,
                stepsize,
                self.n_steps
            )

            new_z, new_stepsize, new_acceptance_rate = hmc_updates(
                z,
                stepsize,
                avg_acceptance_rate=avg_acceptance_rate,
                final_pos=final_pos,
                accept=accept,
                stepsize_min=self.stepsize_min,
                stepsize_max=self.stepsize_max,
                stepsize_dec=self.stepsize_dec,
                stepsize_inc=self.stepsize_inc,
                target_acceptance_rate=self.target_acceptance_rate,
                avg_acceptance_slowness=self.avg_acceptance_slowness
            )

            return index+1, w, new_z, new_stepsize, new_acceptance_rate

        index = 0
        w =torch.zeros([self.num_samples, self.batch_size])
        z = z.float()
        old_z = z.clone()
        stepsize = self.stepsize
        avg_acceptance_rate = self.avg_acceptance_rate
        while index < len(schedule) -1:
            index, w, z, stepsize, avg_acceptance_rate = body(index, w, z, stepsize, avg_acceptance_rate)

        return torch.logsumexp(w,axis=0).squeeze() - torch.log(torch.ones(1) * w.size(0))

