import os
from time import time

import argparse
# import setGPU

import torch
from architectures import get_architecture
from datasets import get_dataset, DATASETS, get_num_classes, get_input_dimension
from distribution import StandardGaussian, GeneralGaussian, LinftyGaussian, LinftyGeneralGaussian, L1GeneralGaussian
from tensorboardX import SummaryWriter

import smooth
from th_heuristic import get_beta, get_beta2

EPS = 1e-5

parser = argparse.ArgumentParser(description='Sampling for Pa')
parser.add_argument("dataset", choices=DATASETS, help="which dataset")
parser.add_argument("base_classifier", type=str, help="path to saved pytorch model of base classifier")
parser.add_argument("std", type=float, help="noise std")
parser.add_argument("--disttype", type=str, help="smoothing distribution type", choices=['gaussian', 'general-gaussian',
                                                                                         'infty-gaussian',
                                                                                         'infty-general-gaussian',
                                                                                         'L1-general-gaussian'])
parser.add_argument("--outbase", type=str, default="data/sampling")
parser.add_argument("--batch", type=int, default=1024, help="batch size")
parser.add_argument("--k", type=int, default=None, help="the parameter for general-gaussian, usually should be close but slightly smaller than d/2")
parser.add_argument("--skip", type=int, default=10, help="how many examples to skip")
parser.add_argument("--start", type=int, default=-1, help="start from max(0, start)")
parser.add_argument("--stop", type=int, default=-1, help="stop when encounter this, i.e., [start, stop)")
parser.add_argument("--split", choices=["train", "test"], default="test", help="train or test set")
parser.add_argument("--N0", type=int, default=100)
parser.add_argument("--N", type=int, default=100000, help="number of samples to use")
parser.add_argument("--alpha", type=float, default=0.0005, help="failure probability")
parser.add_argument("--th", default=1.0, help="Specific for general-gaussian with hard thresholded norm")
parser.add_argument('--gpu', default=None, type=str,
                    help='id(s) for CUDA_VISIBLE_DEVICES')
parser.add_argument('--comment', default=None, type=str, help='special annotation to the model type')
args = parser.parse_args()

if __name__ == '__main__':
    if args.gpu:
        os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

    if not os.path.exists(args.outbase):
        os.makedirs(args.outbase)

    # load the base classifier
    chkp = torch.load(args.base_classifier)
    model = get_architecture(chkp["arch"], args.dataset, args.comment)
    model.load_state_dict(chkp['state_dict'])
    model.eval()

    dataset = get_dataset(args.dataset, args.split)
    num_classes = get_num_classes(args.dataset)
    d = get_input_dimension(args.dataset)

    # obtain the filename cropped by extension
    out_dir = str(os.path.basename(args.base_classifier))
    out_dir = '.'.join(out_dir.split('.')[:-1])
    out_dir = os.path.join(args.outbase, out_dir)
    file_name = f'{args.disttype}-{args.k}-{args.std}-{args.N}-{args.alpha}'
    if isinstance(args.th, str) or abs(args.th - 1.0) > EPS:
        file_name += f'-{args.th}'
    file_name += '.txt'
    print(f'output to {os.path.join(out_dir, file_name)}')

    # init tensorboard writer
    writer = SummaryWriter(os.path.join(out_dir, f'{args.disttype}-{args.k}-{args.std}-{args.N}-{args.alpha}'))

    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    f = open(os.path.join(out_dir, file_name), 'a')

    if not isinstance(args.th, float) and args.th[0].isdigit():
        args.th = float(args.th)

    if args.disttype == 'gaussian':
        distribution = StandardGaussian(d, args.std)
    elif args.disttype == 'general-gaussian':
        distribution = GeneralGaussian(d, args.k, args.std, th=args.th if isinstance(args.th, float) else 1.0)
    elif args.disttype == 'infty-gaussian':
        distribution = LinftyGaussian(d, args.std)
    elif args.disttype == 'infty-general-gaussian':
        distribution = LinftyGeneralGaussian(d, args.k, args.std)
    elif args.disttype == 'L1-general-gaussian':
        distribution = L1GeneralGaussian(d, args.k, args.std)
    else:
        raise NotImplementedError('Unsupported smoothing distribution')

    print(distribution.info())

    """
        Print metainfo
    """
    print(f"""
x disttype {args.disttype}
x k {args.k}
x std {args.std}
x N {args.N}
x alpha {args.alpha}
""", file=f)
    f.flush()
    """
        Finish print metainfo
    """

    """
        Prepare for heuristically select beta
    """
    if isinstance(args.th, str):
        # read old data
        old_pas = dict()
        old_pars = dict()
        old_file_name = f'{args.disttype}-{args.k}-{args.std}-{args.N}-{args.alpha}.txt'
        with open(os.path.join(out_dir, old_file_name), 'r') as fin:
            for line in fin.readlines():
                if line.startswith('o'):
                    things = line.split(' ')
                    old_pas[int(things[1])] = float(things[2])
                    old_pars[int(things[1])] = float(things[3])


    stime = time()
    tot_p1low = 0.
    tot_instance = 0
    for i in range(len(dataset)):

        # only certify every args.skip examples, and stop after args.max examples
        if i % args.skip != 0:
            continue
        if i < args.start:
            continue
        if i == args.stop:
            break
        tot_instance += 1

        (x, label) = dataset[i]
        x = x.cuda()

        sstime = time()

        nA_base, realN_base = 0, 0
        local_alpha = args.alpha

        ##### heuristic region #####
        ##### should be made consistent with the computation function algo.py #####
        if args.th == 'x':
            beta = get_beta(old_pas[i])
            print(f'  old pa = {old_pas[i]} leads to beta = {beta}')
            distribution.set_th(beta)

        if args.th == 'x+':
            if old_pars[i] >= 1.0 - 1e-8:
                print(f'  old pa_r equals to 1, leads to another round of sampling')
                distribution.set_th(1.0)
                # aggregate previous samples
                nA_base += args.N
                realN_base += args.N
                local_alpha *= 2.
            else:
                beta = get_beta(old_pas[i])
                print(f'  old pa = {old_pas[i]} leads to beta = {beta}')
                distribution.set_th(beta)

        if args.th == 'x2':
            beta = get_beta2(old_pas[i])
            print(f'  old pa = {old_pas[i]} leads to beta = {beta}')
            distribution.set_th(beta)

        # a rarely-used setting
        if args.th == 'x2+':
            if old_pars[i] >= 1.0 - 1e-9:
                print(f'  old pa_r equals to 1, leads to another round of sampling')
                distribution.set_th(1.0)
                # aggregate previous samples
                nA_base += args.N
                realN_base += args.N
                local_alpha *= 2.
            else:
                beta = get_beta2(old_pas[i])
                print(f'  old pa = {old_pas[i]} leads to beta = {beta}')
                distribution.set_th(beta)

        # draw more samples of f(x + epsilon)
        nA, realN = smooth.sample_noise(model, x, distribution, label, args.N, num_classes, args.batch)
        # use these samples to estimate a lower bound on pA
        # nA = counts_estimation[label].item()
        nA, realN = nA + nA_base, realN + realN_base
        print(f'   {nA} out of {realN} sampled')
        # confidence interval
        p1low, p1high = smooth.confidence_bound(nA, realN, local_alpha)


        print(f'#{i} [{p1low}, {p1high}] {time() - sstime} s ({time() - stime} s)')
        print(f"o {i} {p1low} {p1high}", file=f)
        f.flush()

        tot_p1low += p1low
        writer.add_scalar('avg-p1low', tot_p1low / tot_instance, i)

    f.close()

