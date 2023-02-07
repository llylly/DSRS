



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

parser = argparse.ArgumentParser(description='EOT attack for smoothed classifier')
parser.add_argument("dataset", choices=DATASETS, help="which dataset")
parser.add_argument("base_classifier", type=str, help="path to saved pytorch model of base classifier")
parser.add_argument("std", type=float, help="noise std")
parser.add_argument("--disttype", type=str, help="smoothing distribution type", choices=['gaussian', 'general-gaussian'])
parser.add_argument("--outbase", type=str, default="data/attack")
parser.add_argument("--k", type=int, default=None, help="the parameter for general-gaussian, usually should be close but slightly smaller than d/2")
parser.add_argument("--skip", type=int, default=10, help="how many examples to skip")
parser.add_argument("--start", type=int, default=-1, help="start from max(0, start)")
parser.add_argument("--stop", type=int, default=-1, help="stop when encounter this, i.e., [start, stop)")
parser.add_argument("--split", choices=["train", "test"], default="test", help="train or test set")
parser.add_argument("--sample", type=int, default=100, help="use how many samples to compute the expected loss")
parser.add_argument("--N", type=int, default=100, help="number of samples to determine correctness")
parser.add_argument("--p", type=str, choices=["2", "inf"], default="2", help="Lp norm type")
parser.add_argument("--eps", type=float, default=0.5, help="perturbation budget")
parser.add_argument("--stepdivider", type=float, default=100, help="step size")
parser.add_argument("--step", type=int, default=200, help="number of steps")
parser.add_argument("--pgd", action='store_true', help="whether to use PGD attack instead of I-FGSM attack. PGD attack starts with a randomly start point")
parser.add_argument("--randomstart", type=int, default=10, help="number of random start. Only available when PGD is enabled")
parser.add_argument('--gpu', default=None, type=str,
                    help='id(s) for CUDA_VISIBLE_DEVICES')
args = parser.parse_args()

"I-FGSM attack"

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
    file_name = f'{args.disttype}-{args.k}-{args.std}-sample-{args.sample}-L-{args.p}-eps-{args.eps}-step-{args.step}'
    if args.pgd:
        file_name += f'-pgd-start-{args.randomstart}'
    stats_file_name = 'stats-' + file_name + '.txt'
    file_name = file_name + '.txt'
    print(f'output to {os.path.join(out_dir, file_name)} and {os.path.join(out_dir, stats_file_name)}')

    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    f = open(os.path.join(out_dir, file_name), 'w')
    sf = open(os.path.join(out_dir, stats_file_name), 'w')

    if args.disttype == 'gaussian':
        distribution = StandardGaussian(d, args.std)
    elif args.disttype == 'general-gaussian':
        distribution = GeneralGaussian(d, args.k, args.std)
    else:
        raise NotImplementedError('Unsupported smoothing distribution')

    print(distribution.info())

    stime = time()
    tot_benign = 0
    tot_robust = 0
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

        # draw samples of f(x + noise)
        counts_selection = smooth.full_sample_noise(model, x, distribution, args.N, num_classes, args.sample)
        # use these samples to take a guess at the top class
        benign_cAHat = counts_selection.argmax().item()
        if benign_cAHat != label:
            print(f'#{i} predicts wrong.')
            can_attack = True
            now_step = 0
            adv_cAHat = benign_cAHat
        else:
            tot_benign += 1

            # now conducting the attack
            noises = distribution.sample(args.sample, cuda=True).reshape([args.sample] + list(x.shape))
            noised_x = noises + x.unsqueeze(0)
            step_size = args.eps / args.stepdivider

            can_attack = False

            start_time = 1 if not args.pgd else args.randomstart

            for s in range(start_time):
                if not args.pgd:
                    # I-FGSM, start from original point
                    delta = torch.zeros_like(x.unsqueeze(0), device='cuda', requires_grad=True)
                else:
                    if args.p == 'inf':
                        delta = torch.rand_like(x.unsqueeze(0), device='cuda', requires_grad=True) * (args.eps * 2.) - args.eps
                    elif args.p == '2':
                        with torch.no_grad():
                            delta = torch.rand_like(x.unsqueeze(0), device='cuda') * (args.eps * 2.) - args.eps
                            if torch.norm(delta) > args.eps:
                                # project to L2 ball
                                delta = delta / torch.norm(delta) * args.eps
                        delta.requires_grad_(True)

                # attack loop
                for now_step in range(args.step):
                    delta = torch.tensor(delta, device='cuda', requires_grad=True)
                    loss = torch.nn.functional.cross_entropy(model(noised_x + delta), torch.full((args.sample,), label, device='cuda'))
                    # print(f'  #{i} loss at step {now_step} = {loss}')
                    loss.backward()

                    with torch.no_grad():
                        direc = delta.grad.detach()
                        if args.p == '2':
                            direc = direc / torch.norm(direc)
                        elif args.p == 'inf':
                            direc = torch.sign(direc)
                        direc = direc * step_size
                        delta = delta + direc

                        if args.p == '2':
                            if torch.norm(delta) > args.eps:
                                # project to L2 ball
                                delta = delta / torch.norm(delta) * args.eps
                            print(f'   #{i} try {s} step {now_step} L2   norm = {torch.norm(delta):.3f}', end='\r', flush=True)
                        elif args.p == 'inf':
                            if torch.max(torch.abs(delta)) > args.eps:
                                # project to Linf ball
                                delta = delta / torch.max(torch.abs(delta)) * args.eps
                            print(f'   #{i} try {s} step {now_step} Linf norm = {torch.max(torch.abs(delta)):.3f}', end='\r', flush=True)

                    counts_selection = smooth.full_sample_noise(model, x + delta, distribution, args.N, num_classes, args.sample)
                    adv_cAHat = counts_selection.argmax().item()
                    if adv_cAHat != label:
                        print(f'#{i} attacked at step {now_step}.')
                        can_attack = True
                        break

                if can_attack:
                    break

            if not can_attack:
                tot_robust += 1

        print(f'#{i} {benign_cAHat == label} {adv_cAHat == label} {time() - sstime} s ({time() - stime} s)')
        print(f"o {i} {benign_cAHat == label} {not can_attack} step {now_step} time {time() - sstime}", file=f)
        print(f'Now benign acc = {tot_benign} / {tot_instance} = {tot_benign / tot_instance}')
        print(f'Now robust acc = {tot_robust} / {tot_instance} = {tot_robust / tot_instance}')
        f.flush()

    print(tot_instance, tot_benign, tot_robust, tot_benign / tot_instance, tot_robust / tot_instance, file=sf)

    f.close()
    sf.close()
    print(f'Benign acc = {tot_benign} / {tot_instance} = {tot_benign / tot_instance}')
    print(f'Robust acc = {tot_robust} / {tot_instance} = {tot_robust / tot_instance}')
