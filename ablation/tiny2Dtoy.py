import os
import numpy as np
from algo.algo import calc_fast_beta_th, check
from distribution import StandardGaussian, GeneralGaussian, LinftyGaussian, LinftyGeneralGaussian, L1GeneralGaussian

# d = 2
# standard Gaussian
# Gaussian variance 1.0
# P_A = 0.8
# what is r for a range of Q_A?

if __name__ == '__main__':
    pA = 0.6
    sigma = 1.0
    d = 784
    k = 380
    bunk_radius_sigma = np.sqrt(d / (d - 2.0 * k)) * sigma


    dist = GeneralGaussian(d=d, k=k, scale=sigma)
    orig_r = 0.2455
    print('orig_r =', orig_r, flush=True)
    with open('data/2D-toy/log.txt', 'a') as f:
        print('orig_r =', orig_r, file=f, flush=True)

    # qAs = [0.70, 0.71, 0.72, 0.73, 0.74,
    #        0.75, 0.76, 0.77, 0.78, 0.79,
    #        0.80, 0.81, 0.82, 0.83, 0.84,
    #        0.85, 0.86, 0.87, 0.88, 0.89,
    #        0.90, 0.91, 0.92, 0.93, 0.94,
    #        0.95, 0.96, 0.97, 0.98, 0.99,
    #        0.999, 0.9999, 0.99999, 0.999999, 1.0]
    # qAs = [0.70,
    #        0.75, #0.76, 0.77, 0.78, 0.79,
    #        0.80, #0.81, 0.82, 0.83, 0.84,
    #        0.85, #0.86, 0.87, 0.88, 0.89,
    #        0.90, #0.91, 0.92, 0.93, 0.94,
    #        0.95, #0.96, 0.97, 0.98, 0.99,
    #        0.999, 0.9999, 0.99999, 0.999999, 1.0]

    qAs = [0.99999, 0.999999, 0.9999999, 0.99999999]

    r = 0.37

    for qA in qAs:
        while True:
            print('checking', r, flush=True)
            if not check('general-gaussian-th', r, d, k, bunk_radius_sigma, 'x+', pA, pA, qA, qA, eps=1e-9):
                break
            r += 0.01
        r -= 0.01
        print('qA =', qA, ' new_r =', r, flush=True)
        with open('data/2D-toy/log.txt', 'a') as f:
            print('qA =', qA, ' new_r =', r, file=f, flush=True)
