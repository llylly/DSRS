import argparse
import time
from typing import Optional

import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Optimizer
from torch.utils.data import DataLoader

from distribution import StandardGaussian, GeneralGaussian, LinftyGaussian, LinftyGeneralGaussian, L1GeneralGaussian

from architectures import ARCHITECTURES
from datasets import DATASETS
from third_party.smoothadv import Attacker
from train_utils import AverageMeter, accuracy, log, requires_grad_
from train_utils import prologue

def init_distribution(k, d, noise_sd, infty=False, L1=False):
    if not infty and not L1:
        if k == 0:
            return StandardGaussian(d, noise_sd)
        else:
            return GeneralGaussian(d, k, noise_sd)
    elif infty:
        if k == 0:
            return LinftyGaussian(d, noise_sd)
        else:
            return LinftyGeneralGaussian(d, k, noise_sd)
    else:
        return L1GeneralGaussian(d, k, noise_sd)

parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
parser.add_argument('dataset', type=str, choices=DATASETS)
parser.add_argument('arch', type=str, choices=ARCHITECTURES)
parser.add_argument('--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--epochs', default=150, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--batch', default=256, type=int, metavar='N',
                    help='batchsize (default: 256)')
parser.add_argument('--lr', '--learning-rate', default=0.1, type=float,
                    help='initial learning rate', dest='lr')
parser.add_argument('--lr_step_size', type=int, default=50,
                    help='How often to decrease learning by gamma.')
parser.add_argument('--gamma', type=float, default=0.1,
                    help='LR is multiplied by gamma on schedule.')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)')
parser.add_argument('--noise_sd', default=0.0, type=float,
                    help="standard deviation of Gaussian noise for data augmentation")
parser.add_argument('--print-freq', default=10, type=int,
                    metavar='N', help='print frequency (default: 10)')
# somehow doesn't work...
# parser.add_argument('--gpu', default=None, type=str,
#                     help='id(s) for CUDA_VISIBLE_DEVICES')

#####################
# Options added by Salman et al. (2019)
parser.add_argument('--resume', action='store_true',
                    help='if true, tries to resume training from existing checkpoint')
parser.add_argument('--pretrained-model', type=str, default='',
                    help='Path to a pretrained model')

#####################
parser.add_argument('--num-noise-vec', default=1, type=int,
                    help="number of noise vectors. `m` in the paper.")
parser.add_argument('--lbd', default=20., type=float)

# Options when SmoothAdv is used (Salman et al., 2019)
parser.add_argument('--adv-training', action='store_true')
parser.add_argument('--epsilon', default=512, type=float)
parser.add_argument('--num-steps', default=4, type=int)
parser.add_argument('--warmup', default=10, type=int, help="Number of epochs over which "
                                                           "the maximum allowed perturbation increases linearly "
                                                           "from zero to args.epsilon.")

parser.add_argument('--k', default=0, type=int, help="Final general Gaussian parameter")
parser.add_argument('--k-warmup', default=100, type=int, help="Number of epochs over which the general Gaussian "
                                                              "parameter increases from zero to desired k")
parser.add_argument('--infty', default=0, type=int, help="whether to use pure infty radial distribution")
parser.add_argument('--mix-infty', default=0, type=int, help="How many batches mix the infty norm noises")
parser.add_argument('--mix-infty-multipler', default=1, type=float, help="The variance ratio between infty and l2")
parser.add_argument('--l1', default=0, type=int, help="whether to use pure L1 radial distribution")

args = parser.parse_args()
if args.adv_training:
    mode = f"salman_{args.epsilon}_{args.num_steps}_{args.warmup}"
elif args.num_noise_vec == 1 or args.lbd < 1e-6:
    mode = f"cohen"
else:
    mode = f"consistency"
if args.infty > 0:
    mode += "/infty"
elif args.l1 > 0:
    mode += '/L1'
elif args.mix_infty > 0:
    mode += f"/mix_infty_{args.mix_infty}"
    if not (1. - 1e-6 < args.mix_infty_multipler < 1. + 1e-6):
        mode += "x" + str(args.mix_infty_multipler)
args.outdir = f"trained_models/{args.dataset}/k_{args.k}_warmup_{args.k_warmup}/{mode}/num_{args.num_noise_vec}/lbd_{args.lbd}/noise_{args.noise_sd}"

args.epsilon /= 256.0

# if args.gpu:
#     os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def kl_div(input, targets, reduction='batchmean'):
    return F.kl_div(F.log_softmax(input, dim=1), targets,
                    reduction=reduction)


def _cross_entropy(input, targets, reduction='mean'):
    targets_prob = F.softmax(targets, dim=1)
    xent = (-targets_prob * F.log_softmax(input, dim=1)).sum(1)
    if reduction == 'sum':
        return xent.sum()
    elif reduction == 'mean':
        return xent.mean()
    elif reduction == 'none':
        return xent
    else:
        raise NotImplementedError()


def _entropy(input, reduction='mean'):
    return _cross_entropy(input, input, reduction)


def main():

    train_loader, test_loader, criterion, model, optimizer, scheduler, \
    starting_epoch, logfilename, model_path, device, writer = prologue(args)

    if args.adv_training:
        attacker = SmoothAdv_PGD(steps=args.num_steps, device=device, max_norm=args.epsilon)
    else:
        attacker = None

    step_counter = {'step': 0}

    for epoch in range(starting_epoch, args.epochs):
        if args.adv_training:
            attacker.max_norm = np.min([args.epsilon, (epoch + 1) * args.epsilon / args.warmup])
        if args.dataset != 'imagenet':
            if args.k == 0:
                now_k = 0
            else:
                now_k = math.ceil(args.k - args.k * math.exp(- epoch * math.log(args.k) / args.k_warmup)) \
                    if epoch <= args.k_warmup else args.k
            print(f'Epoch {epoch} with k = {now_k}')

        before = time.time()
        if args.dataset != 'imagenet':
            train_loss, train_acc = train(train_loader, model, criterion, optimizer, epoch, now_k, args.mix_infty,
                                          args.noise_sd, attacker, device, writer)
        else:
            train_loss, train_acc = train(train_loader, model, criterion, optimizer, epoch, args.k, args.mix_infty,
                                          args.noise_sd, attacker, device, writer, args.k_warmup, step_counter)
        if args.dataset != 'imagenet':
            test_loss, test_acc = test(test_loader, model, criterion, epoch, now_k,
                                       args.noise_sd, device, writer, args.print_freq)
        else:
            if args.k == 0:
                now_k = 0
            else:
                now_k = math.ceil(args.k - args.k * math.exp(- step_counter['step'] * math.log(args.k) / args.k_warmup))\
                    if step_counter['step'] <= args.k_warmup else args.k
            test_loss, test_acc = test(test_loader, model, criterion, epoch, now_k,
                                       args.noise_sd, device, writer, args.print_freq)
        after = time.time()

        log(logfilename, "{}\t{:.3}\t{:.3}\t{:.3}\t{:.3}\t{:.3}\t{:.3}".format(
            epoch, after - before,
            scheduler.get_lr()[0], train_loss, train_acc, test_loss, test_acc))

        # In PyTorch 1.1.0 and later, you should call `optimizer.step()` before `lr_scheduler.step()`.
        # See more details at https://pytorch.org/docs/stable/optim.html#how-to-adjust-learning-rate
        scheduler.step(epoch)

        torch.save({
            'epoch': epoch + 1,
            'arch': args.arch,
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict(),
        }, model_path)


def _chunk_minibatch(batch, num_batches):
    X, y = batch
    batch_size = len(X) // num_batches
    for i in range(num_batches):
        yield X[i*batch_size : (i+1)*batch_size], y[i*batch_size : (i+1)*batch_size]


def train(loader: DataLoader, model: torch.nn.Module, criterion, optimizer: Optimizer, epoch: int, k: int, mix_infty:int,
          noise_sd: float,
          attacker: Attacker, device: torch.device, writer=None, k_warmup=None, step_counter=None):
    """
        If step_counter is not None, the step_counter saves the real step and k stores the k limit.
        Otherwise, k stores the real k
    :param loader:
    :param model:
    :param criterion:
    :param optimizer:
    :param epoch:
    :param k:
    :param mix_infty:
    :param noise_sd:
    :param attacker:
    :param device:
    :param writer:
    :param stepwise_k:
    :param step_counter:
    :return:
    """

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    losses_reg = AverageMeter()
    confidence = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    end = time.time()

    # switch to train mode
    model.train()
    requires_grad_(model, True)

    step_c = step_counter['step'] if step_counter is not None else None
    k_lim = k if step_counter is not None else None

    for i, batch in enumerate(loader):
        # measure data loading time
        data_time.update(time.time() - end)

        distribution = None
        distribution_infty = None

        if step_c is not None:
            # init real k then
            if k_lim == 0:
                k = 0
            else:
                k = math.ceil(k_lim - k_lim * math.exp(- step_c * math.log(k_lim) / k_warmup)) if step_c <= k_warmup else k_lim
            step_c += 1
            step_counter['step'] += 1

        mini_batches = _chunk_minibatch(batch, args.num_noise_vec)
        for inputs, targets in mini_batches:
            inputs, targets = inputs.to(device), targets.to(device)
            batch_size = inputs.size(0)

            if distribution is None:
                d = inputs.reshape(batch_size, -1).size(1)
                distribution = init_distribution(k, d, noise_sd, infty=(args.infty > 0), L1=(args.l1 > 0))
                if mix_infty > 0:
                    distribution_infty = init_distribution(k, d, noise_sd * args.mix_infty_multipler, infty=True)

            noises = [torch.tensor(distribution.sample(batch_size).astype(np.float32), device=device).reshape_as(inputs)
                      for _ in range(args.num_noise_vec - mix_infty)] + \
                     [torch.tensor(distribution_infty.sample(batch_size).astype(np.float32), device=device).reshape_as(inputs)
                                                                         for _ in range(mix_infty)]

            if args.adv_training:
                requires_grad_(model, False)
                model.eval()
                inputs = attacker.attack(model, inputs, targets, noises=noises)
                model.train()
                requires_grad_(model, True)

            # augment inputs with noise
            inputs_c = torch.cat([inputs + noise for noise in noises], dim=0)
            targets_c = targets.repeat(args.num_noise_vec)

            logits = model(inputs_c)

            loss_xent = criterion(logits, targets_c)

            logits_chunk = torch.chunk(logits, args.num_noise_vec, dim=0)
            softmax = [F.softmax(logit, dim=1) for logit in logits_chunk]
            avg_softmax = sum(softmax) / args.num_noise_vec

            consistency = [kl_div(logit, avg_softmax, reduction='none').sum(1)
                           + _entropy(avg_softmax, reduction='none')
                           for logit in logits_chunk]
            consistency = sum(consistency) / args.num_noise_vec
            consistency = consistency.mean()

            loss = loss_xent + args.lbd * consistency

            avg_confidence = -F.nll_loss(avg_softmax, targets)

            acc1, acc5 = accuracy(logits, targets_c, topk=(1, 5))
            losses.update(loss_xent.item(), batch_size)
            losses_reg.update(consistency.item(), batch_size)
            confidence.update(avg_confidence.item(), batch_size)
            top1.update(acc1.item(), batch_size)
            top5.update(acc5.item(), batch_size)

            # compute gradient and do SGD step
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'k {k}\t'
                  'Time {batch_time.avg:.3f}\t'
                  'Data {data_time.avg:.3f}\t'
                  'Loss {loss.avg:.4f}\t'
                  'Acc@1 {top1.avg:.3f}\t'
                  'Acc@5 {top5.avg:.3f}'.format(
                epoch, i, len(loader), k=k, batch_time=batch_time,
                data_time=data_time, loss=losses, top1=top1, top5=top5))

    writer.add_scalar('loss/train', losses.avg, epoch)
    writer.add_scalar('loss/consistency', losses_reg.avg, epoch)
    writer.add_scalar('loss/avg_confidence', confidence.avg, epoch)
    writer.add_scalar('batch_time', batch_time.avg, epoch)
    writer.add_scalar('accuracy/train@1', top1.avg, epoch)
    writer.add_scalar('accuracy/train@5', top5.avg, epoch)
    writer.add_scalar('train/k', k, epoch)

    # store back new k
    if step_counter is not None:
        step_counter['step'] = step_c

    return (losses.avg, top1.avg)


def test(loader, model, criterion, epoch, k, noise_sd, device, writer=None, print_freq=10):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    end = time.time()

    # switch to eval mode
    model.eval()

    with torch.no_grad():
        distribution = None

        for i, (inputs, targets) in enumerate(loader):

            if distribution is None:
                batch_size = inputs.size(0)
                d = inputs.reshape(batch_size, -1).size(1)
                distribution = init_distribution(k, d, noise_sd, infty=(args.infty > 0), L1=(args.l1 > 0))

            # measure data loading time
            data_time.update(time.time() - end)

            inputs, targets = inputs.to(device), targets.to(device)

            # augment inputs with noise
            noise = distribution.sample(inputs.size(0)).astype(np.float32)
            noise = torch.tensor(noise, device=device).reshape_as(inputs)
            inputs = inputs + noise

            # compute output
            outputs = model(inputs)
            loss = criterion(outputs, targets)

            # measure accuracy and record loss
            acc1, acc5 = accuracy(outputs, targets, topk=(1, 5))
            losses.update(loss.item(), inputs.size(0))
            top1.update(acc1.item(), inputs.size(0))
            top5.update(acc5.item(), inputs.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % print_freq == 0:
                print('Test: [{0}/{1}]\t'
                      'Time {batch_time.avg:.3f}\t'
                      'Data {data_time.avg:.3f}\t'
                      'Loss {loss.avg:.4f}\t'
                      'Acc@1 {top1.avg:.3f}\t'
                      'Acc@5 {top5.avg:.3f}'.format(
                    i, len(loader), batch_time=batch_time, data_time=data_time,
                    loss=losses, top1=top1, top5=top5))

        if writer:
            writer.add_scalar('loss/test', losses.avg, epoch)
            writer.add_scalar('accuracy/test@1', top1.avg, epoch)
            writer.add_scalar('accuracy/test@5', top5.avg, epoch)

        return (losses.avg, top1.avg)


class SmoothAdv_PGD(Attacker):
    """
    SmoothAdv PGD L2 attack

    Parameters
    ----------
    steps : int
        Number of steps for the optimization.
    max_norm : float or None, optional
        If specified, the norms of the perturbations will not be greater than this value which might lower success rate.
    device : torch.device, optional
        Device on which to perform the attack.

    """

    def __init__(self,
                 steps: int,
                 random_start: bool = True,
                 max_norm: Optional[float] = None,
                 device: torch.device = torch.device('cpu')) -> None:
        super(SmoothAdv_PGD, self).__init__()
        self.steps = steps
        self.random_start = random_start
        self.max_norm = max_norm
        self.device = device

    def attack(self, model, inputs, labels, noises=None):
        """
        Performs SmoothAdv PGD L2 attack of the model for the inputs and labels.

        Parameters
        ----------
        model : nn.Module
            Model to attack.
        inputs : torch.Tensor
            Batch of samples to attack. Values should be in the [0, 1] range.
        labels : torch.Tensor
            Labels of the samples to attack.
        noises : List[torch.Tensor]
            Lists of noise samples to attack.

        Returns
        -------
        torch.Tensor
            Batch of samples modified to be adversarial to the model.

        """
        if inputs.min() < 0 or inputs.max() > 1: raise ValueError('Input values should be in the [0, 1] range.')

        def _batch_l2norm(x):
            x_flat = x.reshape(x.size(0), -1)
            return torch.norm(x_flat, dim=1)

        adv = inputs.detach()
        alpha = self.max_norm / self.steps * 2
        for i in range(self.steps):
            adv.requires_grad_()
            logits = [model(adv + noise) for noise in noises]

            softmax = [F.softmax(logit, dim=1) for logit in logits]
            avg_softmax = sum(softmax) / len(noises)
            logsoftmax = torch.log(avg_softmax.clamp(min=1e-20))
            loss = F.nll_loss(logsoftmax, labels)

            grad = torch.autograd.grad(loss, [adv])[0]
            grad_norm = _batch_l2norm(grad).view(-1, 1, 1, 1)
            grad = grad / (grad_norm + 1e-8)

            adv = adv + alpha * grad
            eta_x_adv = adv - inputs
            eta_x_adv = eta_x_adv.renorm(p=2, dim=0, maxnorm=self.max_norm)

            adv = inputs + eta_x_adv
            adv = torch.clamp(adv, 0, 1)
            adv = adv.detach()

        return adv


if __name__ == "__main__":
    main()
