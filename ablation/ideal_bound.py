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

# k = 0
# d = 10

k = 1530.
d = 3072.

def grid_gen(k, d, step=0.01, uplim=2.):

    xs = list()
    ys = list()

    eps = 1e-6
    for i in range(int(uplim // step)):
        now_r = (i + 1) * step

        if k == 0:
            # P1 = norm.cdf(now_r)
            # print(f'P1 =', P1)

            T_sq = 2.0 * gamma(d/2).ppf(norm.cdf(now_r))

            # print(f'T^2 = {T_sq}')

            r_l = now_r
            r_r = 100. * uplim
            while r_r - r_l > eps:
                mid = (r_l + r_r) / 2.
                rshift = gamma(d/2).expect(lambda t: beta((d - 1) / 2, (d - 1) / 2).cdf( (T_sq - (np.sqrt(2.0 * t) - mid) ** 2) / (4.0 * mid * np.sqrt(2.0 * t)) ))
                if rshift > 0.5:
                    r_l = mid
                else:
                    r_r = mid
            ans = r_l

            print(f'[{time.time() - stime:.3f} s] {now_r:.3f} -> {ans:.3f}')

            xs.append(now_r)
            ys.append(ans)

        else:

            scaling = d / (d - 2.0 * k)
            sq_scaling = np.sqrt(scaling)

            lb1_L = -3000.
            lb1_R = +3000.
            while lb1_R - lb1_L > eps:
                lb1_mid = (lb1_L + lb1_R) / 2.
                rshift = 1.0 - gamma(d/2 - k).expect(lambda t: beta((d - 1.) / 2, (d - 1.) / 2).cdf(
                    ((now_r + sq_scaling * np.sqrt(2.0 * t)) ** 2 - 2.0 * k * scaling * lambertWlog(t/k + lb1_mid/k + np.log(t/k)))
                    /
                    (4.0 * now_r * sq_scaling * np.sqrt(2.0 * t))
                ))
                if rshift > 0.5:
                    lb1_R = lb1_mid
                else:
                    lb1_L = lb1_mid
                # print(f'[{lb1_L}, {lb1_R}]')

            # print('now_r = ', now_r, 'lb1_mid =', lb1_mid)

            P1 = gamma(d/2-k).expect(lambda t: beta((d - 1.) / 2, (d - 1.) / 2).cdf(
                ((now_r + sq_scaling * np.sqrt(2.0 * t)) ** 2 - 2.0 * k * scaling * lambertWlog(t/k - lb1_mid/k + np.log(t/k)))
                /
                (4.0 * now_r * sq_scaling * np.sqrt(2.0 * t))
            ))
            T_sq = 2.0 * scaling * gamma(d/2 - k).ppf(P1)

            # print('P1 =', P1)

            r_l = now_r
            r_r = 100. * uplim
            while r_r - r_l > eps:
                mid = (r_l + r_r) / 2.
                rshift = gamma(d/2-k).expect(lambda t: beta((d - 1) / 2, (d - 1) / 2).cdf( (T_sq - (sq_scaling * np.sqrt(2.0 * t) - mid) ** 2) / (4.0 * mid * sq_scaling * np.sqrt(2.0 * t)) ))
                if rshift > 0.5:
                    r_l = mid
                else:
                    r_r = mid
            ans = r_l

            print(f'[{time.time() - stime:.3f} s] {now_r:.3f} -> {ans:.3f}')

            xs.append(now_r)
            ys.append(ans)


    return xs, ys

if __name__ == '__main__':
    # xs, ys = grid_gen(k, d)
    # print(xs, ys)
    #
    # plt.plot(xs, ys)
    # plt.show()

    # aggre = dict()
    # nowd = 10
    # while nowd <= 1000000:
    #     nowk = (nowd - 10) / 2
    #     print('now dim =', nowd, 'k =', nowk)
    #     xs, ys = grid_gen(nowk, nowd)
    #     aggre[nowd] = [nowk, xs, ys]
    #     nowd = nowd * 2
    #
    # with open(os.path.join('ablation/ideal_bound_wrt_dim.txt'), 'w') as f:
    #     json.dump(aggre, f, indent=2)

    aggre = dict()
    for nowd in range(2, 52, 2):
        print('now dim =', nowd, 'k =', 0)
        xs, ys = grid_gen(0, nowd)
        aggre[nowd] = [0, xs, ys]

    with open(os.path.join('ablation/ideal_bound_wrt_dim_mode2.txt'), 'w') as f:
        json.dump(aggre, f, indent=2)

