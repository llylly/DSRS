import os
import sys
sys.path.append('..')
import time
import numpy as np
from multiprocessing import Value
from multiprocessing.pool import Pool, ThreadPool
from algo.algo import bin_search_lambda_2_on_P_numerical, Q_double_var_numerical, Pshift_double_var_numerical
from algo.algo import sampler_MC, Pshift_double_var_MC, P_double_var_MC, Q_double_var_MC

workers = 45
disttype = 'gaussian'
d = 3
k = 0
GRID_X = 100
GRID_Y = 20
# GRID_X = 5
# GRID_Y = 3
sigma = 0.50
beta = 0.40
r = 0.50

bisearch_precision=10.
bisearch_boundary=5000.
range_width=22.0
eps = 1e-6

counter = 0
tot = 0

def init(args):
    global counter
    counter = args

def work(args):
    global counter
    global tot
    global samples

    now_x, now_y, q1_for_lambda1_L, q1_for_lambda1_R, lambda_2_for_lambda_1_L, lambda_2_for_lambda_1_R = args
    print(now_x, now_y)

    lambda_1_L = - range_width
    lambda_1_R = + range_width

    if now_x < eps or now_x > 1.0 - eps:
        p1shift = now_x
    else:
        if q1_for_lambda1_L > now_y:
            # search range of lambda2
            lamb1_L, lamb1_R = lambda_1_L, lambda_1_R
            lamb2_L, lamb2_R = lambda_2_for_lambda_1_R, lambda_2_for_lambda_1_L
            while lamb1_R - lamb1_L > eps:
                lamb1_mid = (lamb1_L + lamb1_R) / 2.0
                lamb2_mid = bin_search_lambda_2_on_P_numerical(disttype, d, k, sigma, beta, lamb1_mid, r, now_x,
                                                               bisearch_precision, bisearch_boundary, eps,
                                                               L=lamb2_L, R=lamb2_R)
                q1_mid = Q_double_var_numerical(disttype, d, k, sigma, beta, lamb1_mid, lamb2_mid, r,
                                                bisearch_precision)
                print(f"    ({lamb1_mid}, {lamb2_mid}) Q = {q1_mid}")
                if q1_mid > now_y:
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
                lamb2_mid = bin_search_lambda_2_on_P_numerical(disttype, d, k, sigma, beta, lamb1_mid, r, now_x,
                                                               bisearch_precision, bisearch_boundary, eps,
                                                               L=lamb2_L, R=lamb2_R)
                q1_mid = Q_double_var_numerical(disttype, d, k, sigma, beta, lamb1_mid, lamb2_mid, r,
                                                bisearch_precision)
                print(f"    ({lamb1_mid}, {lamb2_mid}) Q = {q1_mid}")
                if q1_mid < now_y:
                    lamb1_L = lamb1_mid
                    lamb2_R = lamb2_mid
                else:
                    lamb1_R = lamb1_mid
                    lamb2_L = lamb2_mid
            lamb1 = lamb1_L
            lamb2 = lamb2_R

        p1shift_L, p1shift_R = Pshift_double_var_MC(disttype, samples, lamb1, lamb2, bisearch_precision)
        # p1shift = Pshift_double_var_numerical(disttype, d, k, sigma, beta, lamb1, lamb2, r, bisearch_precision,
        #                                       eps=1e-8)
        p1shift = (p1shift_L + p1shift_R) / 2.0

    with counter.get_lock():
        counter.value += 1
        print(f'[{counter.value}/{tot}] ({now_x}, {now_y}) -> {p1shift}')
    return p1shift



if __name__ == '__main__':
    samples = sampler_MC(disttype, d, k, sigma, beta, r, 500000)

    stime = time.time()
    grid_x, grid_y = list(), list()
    grid_q1L, grid_q1R, grid_lamb2L, grid_lamb2R = list(), list(), list(), list()
    ans = list()

    q1L, q1R = dict(), dict()
    lamb2L, lamb2R = dict(), dict()

    for i in range(GRID_X + 1):
        now_x = i / GRID_X
        if now_x > eps and now_x < 1.0 - eps:

            lambda_1_L = - range_width
            lambda_1_R = + range_width
            lambda_2_for_lambda_1_L = bin_search_lambda_2_on_P_numerical(disttype, d, k, sigma, beta, lambda_1_L, r, now_x,
                                                                         bisearch_precision, bisearch_boundary, eps)
            q1_for_lambda1_L = Q_double_var_numerical(disttype, d, k, sigma, beta,
                                                      lambda_1_L, lambda_2_for_lambda_1_L, r,
                                                      bisearch_precision, limit=50)
            # print('-------')
            lambda_2_for_lambda_1_R = bin_search_lambda_2_on_P_numerical(disttype, d, k, sigma, beta, lambda_1_R, r, now_x,
                                                                         bisearch_precision, bisearch_boundary, eps)
            # print(f'lambda_1_r: {lambda_1_R}, lambda_2_r: {lambda_2_for_lambda_1_R} sigma={sigma} beta={beta}')
            q1_for_lambda1_R = Q_double_var_numerical(disttype, d, k, sigma, beta,
                                                      lambda_1_R, lambda_2_for_lambda_1_R, r,
                                                      bisearch_precision, limit=50)
            q1L[now_x] = (q1_for_lambda1_L)
            q1R[now_x] = (q1_for_lambda1_R)
            lamb2L[now_x] = (lambda_2_for_lambda_1_L)
            lamb2R[now_x] = (lambda_2_for_lambda_1_R)

            print(f'for {now_x}: {q1_for_lambda1_L} - {q1_for_lambda1_R}')

    for i in range(GRID_X + 1):
        now_x = i / GRID_X

        if now_x < eps or now_x > 1.0 - eps:
            grid_x.append(now_x)
            grid_y.append(now_x)
            grid_q1L.append(None)
            grid_q1R.append(None)
            grid_lamb2L.append(None)
            grid_lamb2R.append(None)
        else:
            ymin = min(q1L[now_x], q1R[now_x]) + eps
            ymax = max(q1L[now_x], q1R[now_x]) - eps
            for now_y in np.linspace(ymin, ymax, GRID_Y):
                grid_x.append(now_x)
                grid_y.append(now_y)
                grid_q1L.append(q1L[now_x])
                grid_q1R.append(q1R[now_x])
                grid_lamb2L.append(lamb2L[now_x])
                grid_lamb2R.append(lamb2R[now_x])

    tot = len(grid_x)
    counter = Value('i', 0)

    with Pool(processes=workers, initializer=init, initargs=(counter,)) as p:
        ans = p.map(work, zip(grid_x, grid_y, grid_q1L, grid_q1R, grid_lamb2L, grid_lamb2R))

    with open('c-landscape.txt', 'w') as f:
        for i, (grid_x, grid_y) in enumerate(zip(grid_x, grid_y)):
            print(grid_x, grid_y, ans[i], file=f)
    print(f'Time elapsed: {time.time() - stime} s')

    #
    # # print(work((0.8484848484848485, 0.8131313131313131)))
    # # print(work((0.8484848484848485, 0.8636363636363636)))
    #
    #
    # lamb1, lamb2 = (12.477981150150299, -12.179908751609851)
    # p1 = P_double_var_MC(disttype, samples, lamb1, lamb2, bisearch_precision)
    # q1 = Q_double_var_MC(disttype, samples, lamb1, lamb2, bisearch_precision)
    # print(p1, q1)
    # p1shift = Pshift_double_var_MC(disttype, samples, lamb1, lamb2, bisearch_precision)
    # print(p1shift)
    # p1shift = Pshift_double_var_numerical(disttype, d, k, sigma, beta, lamb1, lamb2, r, bisearch_precision,
    #                                       eps=1e-8)
    # print(p1shift)
    #
    # lamb1, lamb2 = (11.27029937505722, -10.59079609665853)
    # p1 = P_double_var_MC(disttype, samples, lamb1, lamb2, bisearch_precision)
    # q1 = Q_double_var_MC(disttype, samples, lamb1, lamb2, bisearch_precision)
    # print(p1, q1)
    # p1shift = Pshift_double_var_MC(disttype, samples, lamb1, lamb2, bisearch_precision)
    # print(p1shift)
    # p1shift = Pshift_double_var_numerical(disttype, d, k, sigma, beta, lamb1, lamb2, r, bisearch_precision,
    #                                       eps=1e-8)
    # print(p1shift)
    #
    # # ans = Pool(workers).map(work, zip(grid_x, grid_y))
    # # with open('c-landscape-v1.txt', 'w') as f:
    # #     for i, (grid_x, grid_y) in enumerate(zip(grid_x, grid_y)):
    # #         print(grid_x, grid_y, ans[i], file=f)
    # # print(f'Time elapsed: {time.time() - stime} s')

