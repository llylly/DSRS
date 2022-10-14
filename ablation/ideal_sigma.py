import scipy
import numpy as np
from scipy.stats import norm
from scipy.stats import gamma, beta
from utils import lambertWlog
import time
import json
import os

from distribution import GeneralGaussian

import matplotlib.pyplot as plt


stime = time.time()


def best_radius(d, k, T, method='new'):

    sigma = 1.0

    if k == 0:
        if method == 'old':
            print(f'!!! pA={gamma(d / 2.0).cdf(T ** 2 / (2.0 * sigma ** 2))}')
            r = sigma * norm.ppf(gamma(d / 2.0).cdf(T ** 2 / (2.0 * sigma ** 2)))
            return r
        elif method == 'new':
            r_l = 0.
            r_r = 10000.
            while r_r - r_l > 1e-6:
                r_mid = (r_l + r_r) / 2.0
                Pshift = gamma(d / 2.0).expect(lambda t: beta((d - 1.0) / 2.0, (d - 1.0) / 2.0).cdf(
                    (T**2 - (sigma * np.sqrt(2.0 * t) - r_mid) ** 2) /
                    (4.0 * r_mid * sigma * np.sqrt(2.0 * t))
                ), limit=500)
                if Pshift > 0.5:
                    r_l = r_mid
                else:
                    r_r = r_mid
            return r_l
    else:
        if method == 'new':
            sigma = sigma * np.sqrt(d / (d - 2.0 * k))
            r_l = 0.
            r_r = 10000.
            while r_r - r_l > 1e-6:
                r_mid = (r_l + r_r) / 2.0
                Pshift = gamma(d / 2.0 - k).expect(lambda t: beta((d - 1.0) / 2.0, (d - 1.0) / 2.0).cdf(
                    (T**2 - (sigma * np.sqrt(2.0 * t) - r_mid) ** 2) /
                    (4.0 * r_mid * sigma * np.sqrt(2.0 * t))
                ))
                if Pshift > 0.5:
                    r_l = r_mid
                else:
                    r_r = r_mid
            return r_l
        elif method == 'old':
            gg = GeneralGaussian(d, k, sigma)
            return gg.certify_radius(gamma(d / 2.0 - k).cdf(T ** 2 / (2.0 * sigma * sigma * d / (d - 2.0 * k))))

compute = False
plot = True

if __name__ == '__main__':
    if compute:
        d = 4
        PA = 0.4

        ans_dict = dict()

        while d <= 2000000:

            T = 1. * np.sqrt(2.0 * gamma(d / 2.0).ppf(PA))

            print(d)
            print('k = 0')
            r = best_radius(d, 0, T, method='old')
            print(r)

            # ans_dict[f'{(d, 0)}'] = dict()
            # ans_dict[f'{(d, 0)}']['old'] = r
            ans_dict[d] = dict()
            ans_dict[d]['gaussian_old'] = r

            r = best_radius(d, 0, T, method='new')
            ans_dict[d]['gaussian_new'] = r

            k = max((d - 10) / 2, 0)
            print('k =', k)
            ans_dict[d]['g_gaussian_k'] = k

            # ans_dict[f'{(d, k)}'] = dict()

            print('old')
            r = best_radius(d, k, T, method='old')
            print(r)

            # ans_dict[f'{(d, k)}']['old'] = r
            ans_dict[d]['g_gaussian_old'] = r

            print('new')
            r = best_radius(d, k, T)
            print(r)

            # ans_dict[f'{(d, k)}']['new'] = r
            ans_dict[d]['g_gaussian_new'] = r

            print('')
            d = (d * 1.5) // 2 * 2

        with open('ablation/ideal_sigma_wrt_dim.txt', 'a') as f:
            json.dump({PA: ans_dict}, f, indent=2)

    if plot:
        with open('ablation/ideal_sigma_wrt_dim.txt', 'r') as f:
            data = json.load(f)
        data = data['0.8']

        X_gaussian_old = list()
        Y_gaussian_old = list()
        X_gaussian_new = list()
        Y_gaussian_new = list()
        X_gg_old = list()
        Y_gg_old = list()
        X_gg_new = list()
        Y_gg_new = list()

        for d in data:
            x = int(float(d))
            item = data[d]
            y_gaussian_old = item['gaussian_old']
            y_gaussian_new = item['gaussian_new']
            y_gg_old = item['g_gaussian_old']
            y_gg_new = item['g_gaussian_new']

            X_gaussian_old.append(x)
            Y_gaussian_old.append(y_gaussian_old)
            if y_gaussian_new > 1e-6:
                X_gaussian_new.append(x)
                Y_gaussian_new.append(y_gaussian_new)
            X_gg_old.append(x)
            Y_gg_old.append(y_gg_old)
            X_gg_new.append(x)
            Y_gg_new.append(y_gg_new)


        plt.clf()
        plt.style.use('seaborn')
        plt.figure(figsize=(8,6))
        plt.subplots_adjust(left=0.13, bottom=0.15, right=0.99, top=0.90, wspace=0, hspace=0)
        plt.ylabel('Certified Radius')
        plt.xlabel('Input Dimension $d$')
        plt.plot(X_gaussian_old, Y_gaussian_old, label='Neyman-Pearson Standard Gaussian')
        plt.plot(X_gaussian_new, Y_gaussian_new, label='DSRS Standard Gaussian')
        plt.plot(X_gg_old, Y_gg_old, label='Neyman-Pearson Standard Gaussian')
        plt.plot(X_gg_new, Y_gg_new, label='DSRS Standard Gaussian')
        plt.xscale('log')
        plt.yscale('log')
        plt.legend()
        plt.title('With $P_A=0.8$ and concentration assumption, the certified radius under different settings')
        plt.show()



