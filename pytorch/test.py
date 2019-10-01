import torch
import numpy as np
import ais
import matplotlib.pyplot as plt
from scipy.stats import norm
import ipdb
import seaborn as sns
sns.set()

class Generator(object):
    def __init__(self, input_dim, output_dim):
        self.input_dim = input_dim
        self.output_dim = output_dim

    def __call__(self, z):
        return z * 2 + 3

def kde_logpdf(x, mu, sigma):
    """
    Calculate the kde logpdf.
    :param x: Shape [num_samples*batch_size, output_dim]
    :param mu: Shape [num_samples*batch_size, output_dim]
    :param sigma: variance
    :return: [num_samples*batch_size]
    """
    # K(u) = 1/sqrt(2*pi) * exp(-0.5 * u^2)
    # p(x) = 1/nh * |sum over n, i|{K((x-xi)/h)}
    # logp(x) = -log(nh) + |sum over n, i|{-log(sqrt(2*pi) + 0.5*((mu-xi)/h)^2}
    # logp(x)[i] = -log(h) -n*log(sqrt(2*pi)) -log(sqrt(2*pi) + 0.5*((mu-xi)/h)^2
    # instead of summing it then taking the average of it, we simply keep things separately in a
    # matrix. log_mean_exp will run in the end of AIS method. and we will obtain a shape
    # [batch_size] instead of [batch_size*num_samples]
#     print(x.)
    d = (x.float() - mu.float())
    d =  d / sigma
    e = -0.5 * torch.pow(d, 2)
    
#     print(mu)
    z = (mu.size(1) *
        torch.log(sigma * np.sqrt(np.pi * 2.0))).float()
    # ipdb.set_trace()
    return e - z


generator = Generator(1, 1)

p = norm()

batch_size = 40
num_samples = 10000
x = np.linspace(norm.ppf(0.01, loc=3, scale=2), norm.ppf(0.99, loc=3, scale=2), batch_size)
p1 = norm.pdf(x, loc=3, scale=2)
x = torch.from_numpy(x)
model = ais.AIS(x, lambda x, z: kde_logpdf(x, generator(z), torch.ones(1)*1.5),
        {'input_dim': 1, 'output_dim': 1, 'batch_size': batch_size},
                  num_samples)

xx = np.reshape(x, [batch_size, 1])

schedule = ais.get_schedule(5, rad=4)

#print(schedule)
target = model.x[...,None,None].repeat(1, model.num_samples, 1).permute(1,0,2)
model.x = target.reshape( 
        model.num_samples * model.batch_size, model.dims['output_dim'])

p2 = model.ais(ais.standard_gaussian([num_samples*batch_size,1]).sample(),
                schedule)
plt.plot(x, p1)
# ipdb.set_trace()
plt.plot(x, np.exp(p2))
plt.savefig('test.png')
