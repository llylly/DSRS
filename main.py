from multiprocessing.pool import Pool, ThreadPool
import argparse
from tensorboardX import SummaryWriter, GlobalSummaryWriter
import os
from time import time, sleep
from random import random
import numpy as np

from scipy.stats import gamma

from datasets import get_input_dimension
from algo.algo import calc_fast_beta_th, check
from utils import read_pAs, read_orig_Rs
from distribution import StandardGaussian, GeneralGaussian, LinftyGaussian, LinftyGeneralGaussian, L1GeneralGaussian


# ====== global settings =======
workers = 10
# workers can be changed by argparse
ORIG_R_EPS = 5e-5
DUAL_EPS = 1e-8
# The above eps works well (and guarantees soundness in practical data) if using argparse's precision
# =============================




dist = None

def orig_radius_pool_func(args):
    """
        Paralleled original radius computing function
    :param args:
    :return:
    """
    no, pA, _ = args
    stime = time()
    print('On #', no)
    r = dist.certify_radius(pA)
    print('#', no, 'done:', time() - stime, 's')
    return r, time() - stime


def calc_orig_radius(base_dir, out_base_dir, disttype, d, k, std, aux_stds, N, alpha):
    """
        The entrance function, or the dispatcher, for the original radius computation
    :param base_dir:
    :param out_base_dir:
    :param disttype:
    :param d:
    :param k:
    :param std:
    :param aux_stds:
    :param N:
    :param alpha:
    :return:
    """
    global dist

    if not os.path.exists(out_base_dir):
        os.makedirs(out_base_dir)

    if disttype == 'gaussian' or disttype == 'gaussian-th':
        dist = StandardGaussian(d, std)
    elif disttype == 'general-gaussian' or disttype == 'general-gaussian-th':
        dist = GeneralGaussian(d, k, std)

    reslist = list()
    restimelist = list()
    if disttype == 'gaussian' or disttype == 'gaussian-th':
        fname = f'gaussian-None-{std}-{N}-{alpha}.txt'
        outname = f'orig-rad-{disttype}-None-{std}-{N}-{alpha}.txt'
        timename = 'time-' + outname
        f = open(os.path.join(out_base_dir, outname), 'a')
        ftime = open(os.path.join(out_base_dir, timename), 'a')
        writer = SummaryWriter(os.path.join(out_base_dir, f'orig-rad-{disttype}-None-{std}-{N}-{alpha}'))

        pAlist = read_pAs(os.path.join(base_dir, fname))
        timestamp = time()
        # list of [no, p1low, p1high]
        for item in pAlist:
            reslist.append((item[0], dist.certify_radius(item[1])))
            print(*(reslist[-1]), file=f)
            f.flush()
            restimelist.append((item[0], time() - timestamp))
            print(*(restimelist[-1]), file=ftime)
            ftime.flush()
            timestamp = time()
    elif disttype == 'general-gaussian' or disttype == 'general-gaussian-th':
        samp_disttype = 'general-gaussian'
        fname = f'{samp_disttype}-{k}-{std}-{N}-{alpha}.txt'
        outname = f'orig-rad-{disttype}-{k}-{std}-{N}-{alpha}.txt'
        timename = 'time-' + outname
        f = open(os.path.join(out_base_dir, outname), 'a')
        ftime = open(os.path.join(out_base_dir, timename), 'a')
        writer = SummaryWriter(os.path.join(out_base_dir, f'orig-rad-{disttype}-{k}-{std}-{N}-{alpha}'))

        pAlist = read_pAs(os.path.join(base_dir, fname))
        res = Pool(workers).map(orig_radius_pool_func, pAlist)
        for i in range(len(res)):
            print('On #', pAlist[i][0])
            now_r, now_time = res[i]
            reslist.append(tuple([pAlist[i][0], now_r]))
            print(*(reslist[-1]), file=f)
            f.flush()
            restimelist.append(tuple([pAlist[i][0], now_time]))
            print(*(restimelist[-1]), file=ftime)
            ftime.flush()

    f.close()
    ftime.close()

    for item in reslist:
        # res: arrary of (no, r)
        writer.add_scalar('orig-radius', item[1], item[0])
    print(reslist)
    return reslist


def combine_info(sampling_dir, orig_rad_dir, disttype, d, k, std, aux_stds, N, alpha):
    """
        info list format:
            Each row is a list - a sample, in ascending order.
            The list format: [no, original R, p1L, p1U, list of other stds]
            The list of other stds follows the order of aux_stds.
            Each item is [p1betaL, p1betaU]
    """
    if disttype == 'general-gaussian' or disttype == 'gaussian':
        orig_rad_fname = f'orig-rad-{disttype}-{k}-{std}-{N}-{alpha}.txt'
        info_list = read_orig_Rs(os.path.join(orig_rad_dir, orig_rad_fname), aux_stds)
        index_mapper = dict()
        for i, item in enumerate(info_list):
            index_mapper[item[0]] = i

        main_pA_fname = f'{disttype}-{k}-{std}-{N}-{alpha}.txt'
        data = read_pAs(os.path.join(sampling_dir, main_pA_fname))
        for item in data:
            if item[0] in index_mapper:
                # p1low
                info_list[index_mapper[item[0]]][2] = item[1]
                # p1high
                info_list[index_mapper[item[0]]][3] = item[2]

        for i, aux_std in enumerate(aux_stds):
            aux_pA_fname = f'{disttype}-{k}-{aux_std}-{N}-{alpha}.txt'
            data = read_pAs(os.path.join(sampling_dir, aux_pA_fname))
            for item in data:
                if item[0] in index_mapper:
                    info_list[index_mapper[item[0]]][4][i][0], info_list[index_mapper[item[0]]][4][i][1] = item[1], item[2]
        return info_list
    elif disttype == 'general-gaussian-th' or disttype == 'gaussian-th':
        orig_rad_fname = f'orig-rad-{disttype[:-3]}-{k}-{std}-{N}-{alpha}.txt'
        info_list = read_orig_Rs(os.path.join(orig_rad_dir, orig_rad_fname), aux_stds)
        index_mapper = dict()
        for i, item in enumerate(info_list):
            index_mapper[item[0]] = i

        main_pA_fname = f'{disttype[:-3]}-{k}-{std}-{N}-{alpha}.txt'
        data = read_pAs(os.path.join(sampling_dir, main_pA_fname))
        for item in data:
            if item[0] in index_mapper:
                # p1low
                info_list[index_mapper[item[0]]][2] = item[1]
                # p1high
                info_list[index_mapper[item[0]]][3] = item[2]

        for i, aux_std in enumerate(aux_stds):
            aux_pA_fname = f'{disttype[:-3]}-{k}-{std}-{N}-{alpha}-{aux_std}.txt'
            data = read_pAs(os.path.join(sampling_dir, aux_pA_fname))
            for item in data:
                if item[0] in index_mapper:
                    info_list[index_mapper[item[0]]][4][i][0], info_list[index_mapper[item[0]]][4][i][1] = item[1], item[2]
        return info_list
    else:
        raise Exception('unsupported disttype')


bunk_disttype = ''
bunk_radius_d = 0
bunk_radius_k = 0
bunk_radius_sigma = 0.
bunk_radius_beta = 0.
bunk_radius_mode = None
bunk_radius_unit = 0.
bunk_radius_eps = 0.

bunk_global_writer = None

def new_radius_pool_func(args):
    no, orig_r, pAsigmaL, pAsigmaR, pAbetaL, pAbetaR = args
    print('On #', no)
    stime = time()

    new_r = orig_r
    if (orig_r <= 1e-5 and pAsigmaL <= 0.1 and pAbetaL <= 0.1) or pAbetaL is None or pAbetaR is None:
        pass
    else:
        if orig_r <= 1e-5:
            print('try even though orig_r is 0')
        if bunk_radius_mode == 'grid':
            slot = int(orig_r / bunk_radius_unit)
            new_r = orig_r
            while True:
                slot += 1
                try:
                    # ! suppress possible exceptions...
                    print(f'check rad = {slot * bunk_radius_unit} for pA = {pAsigmaL} (old rad = {orig_r})')
                    if check(bunk_disttype, slot * bunk_radius_unit,
                             bunk_radius_d, bunk_radius_k, bunk_radius_sigma, bunk_radius_beta,
                             pAsigmaL, pAsigmaR, pAbetaL, pAbetaR,
                             eps=DUAL_EPS):
                        new_r = bunk_radius_unit * slot
                        print(f'  #{no} New r = {new_r}')
                    else:
                        break
                except Exception as e:
                    # print(type(e))
                    print('exception encountered')
                    print(e)
                    break
                    # raise e
        else:
            r_delta = bunk_radius_eps
            while True:
                print(f'  #{no} Check radius +', r_delta)
                if r_delta > 50.0 * orig_r and orig_r >= 0.1:
                    # I don't quite believe DSRS can improve over 5000% in practice (though theoretically can as described in our paper)
                    raise Exception(f'Suspected numerical error @ #{no} with orig R = {orig_r}, pA in [{pAsigmaL}, {pAsigmaR}], pAbeta in [{pAbetaL}, {pAbetaR}]')
                if check(bunk_disttype, orig_r + r_delta,
                         bunk_radius_d, bunk_radius_k, bunk_radius_sigma, bunk_radius_beta,
                         pAsigmaL, pAsigmaR, pAbetaL, pAbetaR, eps=DUAL_EPS):
                    r_delta *= 2.
                else:
                    r_delta /= 2.
                    break
            if r_delta >= bunk_radius_eps:
                new_r = orig_r + r_delta
            if bunk_radius_mode == 'precise':
                rad_L, rad_R = orig_r + r_delta, orig_r + 2. * r_delta
                while rad_R - rad_L > bunk_radius_eps:
                    mid = (rad_L + rad_R) / 2.0
                    res = check(bunk_disttype, mid,
                                bunk_radius_d, bunk_radius_k, bunk_radius_sigma, bunk_radius_beta,
                                pAsigmaL, pAsigmaR, pAbetaL, pAbetaR, eps=DUAL_EPS)
                    if res:
                        rad_L = mid
                    else:
                        rad_R = mid
                new_r = rad_L
    print(f'Result on #{no} (sigma={bunk_radius_sigma}, beta={bunk_radius_beta}) R = {orig_r} + {new_r - orig_r} [time={time() - stime} s]')
    runtime = time() - stime
    # avoid conflict on global writer
    sleep(random())
    bunk_global_writer.add_scalar(f'new-radius({bunk_radius_mode}-mode)', new_r)
    return no, new_r, new_r - orig_r, ((new_r - orig_r) / max(orig_r, 1e-5)), runtime


def bunk_radius_calc(full_info, result_dir, disttype, d, k, sigma, betas, N, alpha, mode, unit, eps):
    """
        The entrance function, or the dispatcher, for the improved radius computation
        :param full_info: the full info list, [[no, radius, p1low, p1high, [[p2low, p2high], ...]]
        :param result_dir: the directory to save the result
        :param d: input dimension
        :param k: for general Gaussian, parameter k; for others, it is None
        :param sigma: variance scaling
        :param betas: for general Gaussian or standard Gaussian, the list of betas to derive improved radius; for others, it is an empty list
        :param N: number of samples
        :param alpha: confidence level
        :param mode: must be 'grid'/'fast'/'precise'
        :param unit: grid search granularity
        :param eps: precision control
        :return: None
    """
    global bunk_disttype, bunk_radius_d, bunk_radius_k, bunk_radius_sigma, bunk_radius_beta
    global bunk_radius_mode, bunk_radius_unit, bunk_radius_eps
    global bunk_global_writer

    if not os.path.exists(result_dir):
        os.makedirs(result_dir)
    print('output to ' + result_dir)
    bunk_disttype = disttype
    bunk_radius_d, bunk_radius_k = d, k
    bunk_radius_mode, bunk_radius_unit, bunk_radius_eps = mode, unit, eps
    if bunk_disttype == 'general-gaussian' or bunk_disttype == 'gaussian' \
            or bunk_disttype == 'general-gaussian-th' or bunk_disttype == 'gaussian-th':
        for i, beta in enumerate(betas):
            print(f'Now on beta = {beta}')
            # compute the real beta
            if bunk_disttype == 'general-gaussian':
                bunk_radius_sigma = np.sqrt(d / (d - 2.0 * k)) * sigma
                bunk_radius_beta = np.sqrt(d / (d - 2.0 * k)) * beta
            elif bunk_disttype == 'general-gaussian-th':
                bunk_radius_sigma = np.sqrt(d / (d - 2.0 * k)) * sigma
                bunk_radius_beta = beta
            else:
                bunk_radius_sigma = sigma
                bunk_radius_beta = beta
            fname = f'new-rad-{disttype}-{k}-{sigma}-{beta}-{N}-{alpha}.txt'
            ftimename = 'time-' + fname
            f = open(os.path.join(result_dir, fname), 'a')
            ftime = open(os.path.join(result_dir, ftimename), 'a')
            bunk_global_writer = GlobalSummaryWriter(os.path.join(result_dir, f'new-rad-{disttype}-{k}-{sigma}-{beta}-{N}-{alpha}'))
            view = [[item[0], item[1], item[2], item[3], item[4][i][0], item[4][i][1]] for item in full_info]
            res = Pool(workers).map(new_radius_pool_func, view)
            [print(item) for item in res]
            [print(*(item[:-1]), file=f) for item in res]
            [print(item[-1], file=ftime) for item in res]
            f.close()
            ftime.close()


parser = argparse.ArgumentParser(description='The main script for DSRS')
parser.add_argument('dataset', choices=['mnist', 'cifar10', 'imagenet', 'tinyimagenet'],
                    description='We rely on the dataset information to obtain the input dimension.'
                                'New datasets can be easily integrated.')
parser.add_argument('task', choices=['origin', 'improved'],
                    description='The workflow is to first use the classical Nayman-Pearson certification to compute the radius -- choices = origin,'
                                'then use the DSRS to compute the improved radius -- choices = improved.'
                                'They are separate runs')
parser.add_argument('model', type=str,
                    description='We load the sampling probability and old radius from the current folder: sampling_dir/model, original_rad_dir/model,'
                                'and save the result to new_rad_dir/model')
parser.add_argument('disttype', choices=['general-gaussian', 'gaussian',
                                         'general-gaussian-th', 'gaussian-th'],
                    description='general-gaussian: generalized Gaussian as P, generalized Gaussian with a different variance as Q;'
                                'gaussian: standard Gaussian as P, standard Gaussian with a different variance as Q (this option obtains invisible improvements when input dimension >= 40);'
                                'general-gaussian-th: generalized Gaussian as P, generalzied Gaussian with threshold cutting as Q;'
                                'gaussian-th: standard Gaussian as P, generalzied Gaussian with threshold cutting as Q (this option obtains invisible improvements when input dimension >= 40).')
parser.add_argument('sigma', type=float,
                    description="sigma of P distribution. Note: as recorded in the paper, here it is sigma instead of sigma'")
parser.add_argument('N', type=int,
                    description='Sampling number. This parameter does not make a difference to the computational method itself. '
                                'It is just for extracting the correct sampling info file whose name includes the sampling number information.')
parser.add_argument('alpha', type=float,
                    description='The certification confidence. Similar to N, this parameter does not make a difference to the computational method itself')
parser.add_argument('-b', action='append', nargs='+',
                    description='The sigma of Q distribution (if using general-gaussian or gaussian as disttype), or '
                                'the percentile of thresholding if a real number between 0 and 1 or percentitle selection heuristics if x, x2, x+, or x2+ (if using general-gaussian-th or gaussian-th as disttype)'
                                '+ insteads for heuristic that includes the fall-back strategy. x+ has the best performance empirically.'
                                'Multiple options can be specified at the same time, and the script will run the certification for them respectively.')

parser.add_argument("--k", type=int, default=None,
                    description="the generalized Gaussian's parameter k. See the paper for the detail")
parser.add_argument('--sampling_dir', type=str, default='data/sampling',
                    description="folder for extracting the sampling pA info.")
parser.add_argument('--original_rad_dir', type=str, default='data/orig-radius',
                    description="folder for extracting (if task = improved) or outputing (if task = origin) the original radius computed by Neyman-Pearson")
parser.add_argument('--new_rad_dir', type=str, default='data/new-radius',
                    description="folder for outputing the new radius compuated by DSRS")

parser.add_argument('--improve_mode', choices=['grid', 'fast', 'precise'], default='grid',
                    description='the strategy for trying the new radius. Empirically, the grid strategy is sufficient if we do not need the precise new radius.')
parser.add_argument('--improve_unit', type=float, default=0.05)
parser.add_argument('--improve_eps', type=float, default=0.0001)

parser.add_argument('--workers', type=int, default=10,
                    description='# processes for parallelized mapping on CPU.')
parser.add_argument('--core_affinity', type=str, default='')
args = parser.parse_args()

if __name__ == '__main__':
    if args.core_affinity is not None and len(args.core_affinity) > 0:
        os.system("taskset -p -c %s %d" % (args.core_affinity, os.getpid()))

    workers = args.workers

    d = get_input_dimension(args.dataset)

    # init for fast Gaussian computation
    calc_fast_beta_th(d)

    k = args.k
    sigma = args.sigma
    if args.b is not None:
        betas = args.b[0]
    else:
        betas = list()
    N = args.N
    alpha = args.alpha

    betas = [float(b) if isinstance(b, str) and b[0].isdigit() else b for b in betas]

    print(f"""
==============
Metainfo:
    task = {args.task}
    model = {args.model}
    distype = {args.disttype}
    d = {d}
    k = {k}
    sigma = {sigma}
    betas = {betas}
    N = {N}
    alpha = {alpha}
==============
""")

    if args.task == 'origin':
        print('Selected task: compute the original R')
        calc_orig_radius(os.path.join(args.sampling_dir, args.model), os.path.join(args.original_rad_dir, args.model),
                         args.disttype, d, k, sigma, betas, N, alpha)
    elif args.task == 'improved':
        print('Selected task: compute the improved R')
        full_info = combine_info(os.path.join(args.sampling_dir, args.model), os.path.join(args.original_rad_dir, args.model),
                                 args.disttype, d, k, sigma, betas, N, alpha)

        bunk_radius_calc(full_info, os.path.join(args.new_rad_dir, args.model),
                         args.disttype, d, k, sigma, betas, N, alpha,
                         args.improve_mode, args.improve_unit, args.improve_eps)

