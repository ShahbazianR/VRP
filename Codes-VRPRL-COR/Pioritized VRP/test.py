import numpy as np
from scipy.stats import truncnorm 

def normal_dist(x, mean=0, sd=2):
    prob_density = (np.pi*sd) * np.exp(-0.5*((x-mean)/sd)**2)
    return prob_density


x = normal_dist(250)
base = normal_dist(0)
print(x, x/base)
