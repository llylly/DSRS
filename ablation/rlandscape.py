import os
import sys
sys.path.append('..')

import time
import numpy as np
from multiprocessing import Value
from multiprocessing.pool import Pool, ThreadPool

from scipy.stats import norm

from algo.algo import bin_search_lambda_2_on_P_numerical, Q_double_var_numerical
from algo.algo import sampler_MC, Pshift_double_var_MC

eps = 1e-6

workers = 10
disttype = 'gaussian'
d = 3
k = 0
GRID_X = 50
GRID_Y = 20
# GRID_X = 5
# GRID_Y = 3
sigma = 0.50
beta = 0.40

r_unit = 0.05
r_precision = 0.01
r_upper = 2.00


bisearch_precision=10.
bisearch_boundary=5000.
range_width=22.0

counter = 0
tot = 0

def init(args):
    global counter
    counter = args

def work(args):
    global counter
    global tot

    lamb2Ls = dict()
    lamb2Rs = dict()
    sampless = dict()

    def check(rad, p1, q1, q1_for_lambda1_L, q1_for_lambda1_R):
        lambda_1_L = - range_width
        lambda_1_R = + range_width
        if rad in lamb2Ls:
            lambda_2_for_lambda_1_L = lamb2Ls[rad]
            lambda_2_for_lambda_1_R = lamb2Rs[rad]
        else:
            lambda_2_for_lambda_1_L = bin_search_lambda_2_on_P_numerical(disttype, d, k, sigma, beta, lambda_1_L, rad, p1,
                                                                         bisearch_precision, bisearch_boundary, eps)
            lambda_2_for_lambda_1_R = bin_search_lambda_2_on_P_numerical(disttype, d, k, sigma, beta, lambda_1_R, rad, p1,
                                                                         bisearch_precision, bisearch_boundary, eps)
            lamb2Ls[rad] = lambda_2_for_lambda_1_L
            lamb2Rs[rad] = lambda_2_for_lambda_1_R

        if q1_for_lambda1_L > q1:
            # search range of lambda2
            lamb1_L, lamb1_R = lambda_1_L, lambda_1_R
            lamb2_L, lamb2_R = lambda_2_for_lambda_1_R, lambda_2_for_lambda_1_L
            while lamb1_R - lamb1_L > eps:
                lamb1_mid = (lamb1_L + lamb1_R) / 2.0
                lamb2_mid = bin_search_lambda_2_on_P_numerical(disttype, d, k, sigma, beta, lamb1_mid, rad, p1,
                                                               bisearch_precision, bisearch_boundary, eps,
                                                               L=lamb2_L, R=lamb2_R)
                q1_mid = Q_double_var_numerical(disttype, d, k, sigma, beta, lamb1_mid, lamb2_mid, rad,
                                                bisearch_precision)
                print(f"    ({lamb1_mid:.3f}, {lamb2_mid:.3f}) Q = {q1_mid:.3f}")
                if q1_mid > q1:
                    lamb1_L = lamb1_mid
                    lamb2_R = lamb2_mid
                else:
                    lamb1_R = lamb1_mid
                    lamb2_L = lamb2_mid
            lamb1 = lamb1_L
            lamb2 = lamb2_R
        else:
            # search range of lambda2
            lamb1_L, lamb1_R = lambda_1_L, lambda_1_R
            lamb2_L, lamb2_R = lambda_2_for_lambda_1_R, lambda_2_for_lambda_1_L
            while lamb1_R - lamb1_L > eps:
                lamb1_mid = (lamb1_L + lamb1_R) / 2.0
                lamb2_mid = bin_search_lambda_2_on_P_numerical(disttype, d, k, sigma, beta, lamb1_mid, rad, p1,
                                                               bisearch_precision, bisearch_boundary, eps,
                                                               L=lamb2_L, R=lamb2_R)
                q1_mid = Q_double_var_numerical(disttype, d, k, sigma, beta, lamb1_mid, lamb2_mid, rad,
                                                bisearch_precision)
                print(f"    ({lamb1_mid:.3f}, {lamb2_mid:.3f}) Q = {q1_mid}")
                if q1_mid < q1:
                    lamb1_L = lamb1_mid
                    lamb2_R = lamb2_mid
                else:
                    lamb1_R = lamb1_mid
                    lamb2_L = lamb2_mid
            lamb1 = lamb1_L
            lamb2 = lamb2_R

        if rad not in sampless:
            samples = sampler_MC(disttype, d, k, sigma, beta, rad, 50000, True, False)
            sampless[rad] = samples
        else:
            samples = sampless[rad]
        p1shift_L, p1shift_R = Pshift_double_var_MC(disttype, samples, lamb1, lamb2, bisearch_precision)
        print(f'    p1shift = {(p1shift_L + p1shift_R) / 2.0:.3f}')

        if (p1shift_L + p1shift_R) / 2.0 > 0.5:
            return True
        else:
            return False


    p1 = args
    print('On', '%.2f' % p1)

    if p1 < eps:
        ans = [(p1, p1, norm.ppf(p1))]
    else:
        ans = [None for _ in range(GRID_Y)]
        # get range of q1
        # the range should be independent of chosen r, so I choose an arbitrary r
        r_tmp = sigma

        lambda_1_L = - range_width
        lambda_1_R = + range_width
        lambda_2_for_lambda_1_L = bin_search_lambda_2_on_P_numerical(disttype, d, k, sigma, beta, lambda_1_L, r_tmp, p1,
                                                                     bisearch_precision, bisearch_boundary, eps)
        q1_for_lambda1_L = Q_double_var_numerical(disttype, d, k, sigma, beta,
                                                  lambda_1_L, lambda_2_for_lambda_1_L, r_tmp,
                                                  bisearch_precision, limit=50)
        # print('-------')
        lambda_2_for_lambda_1_R = bin_search_lambda_2_on_P_numerical(disttype, d, k, sigma, beta, lambda_1_R, r_tmp, p1,
                                                                     bisearch_precision, bisearch_boundary, eps)
        # print(f'lambda_1_r: {lambda_1_R}, lambda_2_r: {lambda_2_for_lambda_1_R} sigma={sigma} beta={beta}')
        q1_for_lambda1_R = Q_double_var_numerical(disttype, d, k, sigma, beta,
                                                  lambda_1_R, lambda_2_for_lambda_1_R, r_tmp,
                                                  bisearch_precision, limit=50)

        q1L, q1U = min(q1_for_lambda1_L, q1_for_lambda1_R), max(q1_for_lambda1_L, q1_for_lambda1_R)
        r_lowbound = norm.ppf(p1) * sigma
        q1M = norm.cdf(r_lowbound / beta)
        print(f'for {p1}: {q1L} - {q1M} - {q1U}')

        q1_cands = np.linspace(q1L + eps, q1U - eps, num=GRID_Y - 1)
        divider = len(q1_cands[q1_cands <= q1M])

        now_rl = r_lowbound
        for i in range(divider - 1, -1, -1):
            print('%.2f' % p1, f'[{divider-i}/{GRID_Y}]', q1_cands[i])

            rp = now_rl + r_unit
            while check(rp, p1, q1_cands[i], q1_for_lambda1_L, q1_for_lambda1_R):
                rp += r_unit
            rn = rp - r_unit
            while rp - rn > r_precision:
                rm = (rn + rp) / 2.
                if check(rm, p1, q1_cands[i], q1_for_lambda1_L, q1_for_lambda1_R):
                    rn = rm
                else:
                    rp = rm
            ans[i] = (p1, q1_cands[i], rn)
            now_rl = rn

            print('%.2f' % p1, f'[{divider-i}/{GRID_Y}]', q1_cands[i], '->', rn)

        ans[divider] = (p1, q1M, r_lowbound)

        now_rl = r_lowbound
        for i in range(divider, len(q1_cands)):
            print('%.2f' % p1, f'[{i + 2}/{GRID_Y}]', q1_cands[i])

            rp = now_rl + r_unit
            while check(rp, p1, q1_cands[i], q1_for_lambda1_L, q1_for_lambda1_R):
                rp += r_unit
            rn = rp - r_unit
            while rp - rn > r_precision:
                rm = (rn + rp) / 2.
                if check(rm, p1, q1_cands[i], q1_for_lambda1_L, q1_for_lambda1_R):
                    rn = rm
                else:
                    rp = rm
            ans[i + 1] = (p1, q1_cands[i], rn)
            now_rl = rn

            print('%.2f' % p1, f'[{i + 2}/{GRID_Y}]', q1_cands[i], '->', rn)


    with counter.get_lock():
        counter.value += 1
        print(f'[{counter.value}/{tot}] {p1} done')
    return ans

if __name__ == '__main__':
    stime = time.time()

    tot = GRID_X
    counter = Value('i', 0)

    ans = list()
    # for item in np.linspace(0.5, 1.0 - (1.0 - 0.5) / GRID_X, num=GRID_X):
    #     ans.append(work(item))
    with Pool(processes=workers, initializer=init, initargs=(counter,)) as p:
        ans = p.map(work, np.linspace(0.5, 1.0 - (1.0 - 0.5) / GRID_X, num=GRID_X))

    print(ans)
    with open('r-landscape.txt', 'w') as f:
        for item in ans:
            for iitem in item:
                print(*iitem, file=f)

    print('tot time', time.time() - stime)
