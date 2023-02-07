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

parser = argparse.ArgumentParser(description='Sampling for Pa')
parser.add_argument("dataset", choices=DATASETS, help="which dataset")
parser.add_argument("base_classifier", type=str, help="path to saved pytorch model of base classifier")
parser.add_argument("std", type=float, help="noise std")
parser.add_argument("--disttype", type=str, help="smoothing distribution type", choices=['gaussian', 'general-gaussian',
                                                                                         'infty-gaussian',
                                                                                         'infty-general-gaussian',
                                                                                         'L1-general-gaussian'])
parser.add_argument("--outbase", type=str, default="data/benign-sampling")
parser.add_argument("--batch", type=int, default=1024, help="batch size")
parser.add_argument("--k", type=int, default=None, help="the parameter for general-gaussian, usually should be close but slightly smaller than d/2")
parser.add_argument("--skip", type=int, default=10, help="how many examples to skip")
parser.add_argument("--start", type=int, default=-1, help="start from max(0, start)")
parser.add_argument("--stop", type=int, default=-1, help="stop when encounter this, i.e., [start, stop)")
parser.add_argument("--split", choices=["train", "test"], default="test", help="train or test set")
parser.add_argument("--N0", type=int, default=100)
parser.add_argument("--N", type=int, default=100000, help="number of samples to use")
parser.add_argument("--alpha", type=float, default=0.0005, help="failure probability")
parser.add_argument('--gpu', default=None, type=str,
                    help='id(s) for CUDA_VISIBLE_DEVICES')
args = parser.parse_args()

if __name__ == '__main__':
    if args.gpu:
        os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

    if not os.path.exists(args.outbase):
        os.makedirs(args.outbase)

    # load the base classifier
    chkp = torch.load(args.base_classifier)
    model = get_architecture(chkp["arch"], args.dataset)
    model.load_state_dict(chkp['state_dict'])
    model.eval()

    dataset = get_dataset(args.dataset, args.split)
    num_classes = get_num_classes(args.dataset)
    d = get_input_dimension(args.dataset)

    # obtain the filename cropped by extension
    out_dir = str(os.path.basename(args.base_classifier))
    out_dir = '.'.join(out_dir.split('.')[:-1])
    out_dir = os.path.join(args.outbase, out_dir)
    file_name = f'{args.disttype}-{args.k}-{args.std}-{args.N}-{args.alpha}.txt'
    print(f'output to {os.path.join(out_dir, file_name)}')


    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    f = open(os.path.join(out_dir, file_name), 'a')

    if args.disttype == 'gaussian':
        distribution = StandardGaussian(d, args.std)
    elif args.disttype == 'general-gaussian':
        distribution = GeneralGaussian(d, args.k, args.std)
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

    stime = time()
    tot_benign = 0.
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

        # draw samples of f(x + epsilon)
        counts_selection = smooth.full_sample_noise(model, x, distribution, args.N0, num_classes, args.batch)
        # use these samples to take a guess at the top class
        cAHat = counts_selection.argmax().item()
        if cAHat != label:
            print(f'#{i} predicts wrong.')
        else:
            tot_benign += 1

        # # draw more samples of f(x + epsilon)
        # counts_estimation = smooth.sample_noise(model, x, distribution, args.N, num_classes, args.batch)
        # # use these samples to estimate a lower bound on pA
        # nA = counts_estimation[label].item()
        # # confidence interval
        # p1low, p1high = smooth.confidence_bound(nA, args.N, args.alpha)


        print(f'#{i} {cAHat == label} {time() - sstime} s ({time() - stime} s)')
        print(f"o {i} {cAHat == label}", file=f)
        f.flush()


    f.close()
    print(f'Benign acc = {tot_benign} / {tot_instance} = {tot_benign / tot_instance}')
