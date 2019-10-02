import matplotlib.pylab as plt
import numpy as np
from scipy.stats import norm
import torch
import ipdb


def kinetic_energy(v):
    return 0.5 * torch.sum(v*v,1)

def hamiltonian(p, v, f):
    return f(p) + kinetic_energy(v)

def metropolis_hastings_accept(energy_prev, energy_next):
    ediff = energy_prev - energy_next
    return (torch.exp(ediff) - torch.rand_like(ediff)) >= 0.0

def simulate_dynamics(initial_pos, initial_vel, stepsize, n_steps, energy_fn):
    
    def get_grad(pos): 
        # TODO(jackson): this seems a bit slow?
        pos = torch.autograd.Variable(pos, requires_grad=True)
        return torch.autograd.grad(energy_fn(pos).sum(), [pos])[0]
    
    def leapfrog(pos, vel, step, i):
        dE_dpos = get_grad(pos)
        new_vel = vel - step * dE_dpos
        new_pos = pos + step * new_vel

        return [new_pos, new_vel, step, i+1]


    dE_dpos = get_grad(initial_pos)
    vel_half_step = initial_vel - 0.5 * stepsize * dE_dpos #
    pos_full_step = initial_pos + stepsize * vel_half_step #

    i = 0
    while i < n_steps:
        pos_full_step, vel_half_step, stepsize, i = leapfrog(pos_full_step, vel_half_step, stepsize, i)
    final_pos = pos_full_step
    new_vel = vel_half_step
    dE_dpos = get_grad(final_pos)
    final_vel = new_vel - 0.5 * stepsize * dE_dpos
    return final_pos, final_vel

def hmc_move(initial_pos, energy_fn, stepsize, n_steps):
    initial_vel = torch.randn_like(initial_pos)
    final_pos, final_vel = simulate_dynamics(
        initial_pos=initial_pos,
        initial_vel=initial_vel,
        stepsize=stepsize,
        n_steps=n_steps,
        energy_fn=energy_fn
    )
    accept = metropolis_hastings_accept(
        energy_prev=hamiltonian(initial_pos, initial_vel, energy_fn),
        energy_next=hamiltonian(final_pos, final_vel, energy_fn)
    )
    return accept, final_pos, final_vel


def hmc_updates(initial_pos, stepsize, avg_acceptance_rate, final_pos, accept,
                target_acceptance_rate, stepsize_inc, stepsize_dec,
                stepsize_min, stepsize_max, avg_acceptance_slowness):
    new_pos = torch.where(accept[:,None], final_pos, initial_pos)
    new_stepsize_ = stepsize*(stepsize_inc if avg_acceptance_rate > target_acceptance_rate else stepsize_dec)

    new_stepsize = np.max([np.min([new_stepsize_, stepsize_max]), stepsize_min])
    new_acceptance_rate = (avg_acceptance_slowness * avg_acceptance_rate +
                                 (1.0 - avg_acceptance_slowness) * torch.mean(accept.float()))
    return new_pos, new_stepsize, new_acceptance_rate

