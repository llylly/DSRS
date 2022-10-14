import os
import sys
sys.path.append('.')
sys.path.append('..')
sys.path.append('../..')
from time import time

import argparse
# import setGPU

from math import ceil
import numpy as np
import torch
from architectures import get_architecture
from datasets import get_dataset, DATASETS, get_num_classes, get_input_dimension
from distribution import sample_l2_vec

import smooth

parser = argparse.ArgumentParser(description='Sampling for Pa')
parser.add_argument("dataset", choices=DATASETS, help="which dataset")
parser.add_argument("base_classifier", type=str, help="path to saved pytorch model of base classifier")
parser.add_argument("start_len", type=float, help="start of magnitude")
parser.add_argument("end_len", type=float, help="end of magnitude")
parser.add_argument("step", type=int, help="query steps")
parser.add_argument("--outbase", type=str, default="data/landscape_sampling")
parser.add_argument("--batch", type=int, default=1024, help="batch size")
parser.add_argument("--skip", type=int, default=10, help="how many examples to skip")
parser.add_argument("--start", type=int, default=-1, help="start from max(0, start)")
parser.add_argument("--stop", type=int, default=-1, help="stop when encounter this, i.e., [start, stop)")
parser.add_argument("--split", choices=["train", "test"], default="test", help="train or test set")
parser.add_argument("--N", type=int, default=100000, help="number of samples to use")
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
    file_name = f'{args.start_len}-{args.end_len}-{args.step}-{args.N}.txt'
    print(f'output to {os.path.join(out_dir, file_name)}')


    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    f = open(os.path.join(out_dir, file_name), 'a')

    stime = time()
    for i in range(len(dataset)):

        # only certify every args.skip examples, and stop after args.max examples
        if i % args.skip != 0:
            continue
        if i < args.start:
            continue
        if i == args.stop:
            break

        (x, label) = dataset[i]
        x = x.cuda()

        sstime = time()

        # # draw samples of f(x + epsilon)
        # counts_selection = smooth.sample_noise(model, x, distribution, args.N0, num_classes, args.batch)
        # # use these samples to take a guess at the top class
        # cAHat = counts_selection.argmax().item()
        # if cAHat != label:
        #     print(f'#{i} predicts wrong.')
        #     continue

        for now_len in np.linspace(start=args.start_len, stop=args.end_len, num=args.step):

            with torch.no_grad():
                counts = 0
                num = args.N
                for j in range(ceil(args.N / args.batch)):
                    print('batch', j, 'num', num, end='\r', flush=True)
                    this_batch_size = min(args.batch, num)
                    num -= this_batch_size

                    batch = x.repeat((this_batch_size, 1, 1, 1))
                    # noise = dist.sample(this_batch_size).astype(np.float32)
                    noise = sample_l2_vec(d, this_batch_size, cuda=True) * now_len
                    noise = torch.tensor(noise, device='cuda').resize_as(batch)
                    predictions = model(batch + noise).argmax(1)
                    counts += torch.sum(predictions == label)
                    # print(noise)
                    # print(predictions)
                    # print(f'label={label}')
                    # print(f'counts={counts}')

            p1 = float(counts) / args.N
            if now_len < 1e-6 and p1 == 0.:
                print('==0 skip')
                break

            # # draw more samples of f(x + epsilon)
            # counts_estimation = smooth.sample_noise(model, x, distribution, args.N, num_classes, args.batch)
            # # use these samples to estimate a lower bound on pA
            # nA = counts_estimation[label].item()
            # # confidence interval
            # p1low, p1high = smooth.confidence_bound(nA, args.N, args.alpha)

            print(f'#{i} [len={now_len:.3f}] {p1:.4f} {time() - sstime} s ({time() - stime} s)')
            print(f"o {i} {now_len} {p1}", file=f)
            f.flush()


    f.close()

