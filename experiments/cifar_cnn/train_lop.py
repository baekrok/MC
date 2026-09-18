import os, sys, shutil, time, random
import argparse
import torch
import torch.backends.cudnn as cudnn
from utils import *
from models import resnet, vgg
import numpy as np
from data_lop import load_data
from torch.utils.data import random_split, ConcatDataset, DataLoader
import wandb
########################################################################################################################
#  Training Baseline
########################################################################################################################
os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("MKL_NUM_THREADS", "4")
torch.set_num_threads(4)

parser = argparse.ArgumentParser(description='Trains ResNet on CIFAR',
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--data_path', type=str, default='./data', help='Path to dataset')
parser.add_argument('--dataset', type=str, default='cifar10',choices=['cifar10', 'cifar100', 'tiny-imagenet'])
parser.add_argument('--arch', type=str, default='resnet18')
parser.add_argument('--train_type', type=str, default='warm',choices=['warm', 'cold'])

# Optimization options
parser.add_argument('--epochs', type=int, default=200, help='Number of epochs to train.')
parser.add_argument('--batch-size', type=int, default=128, help='Batch size.')
parser.add_argument('--learning-rate', type=float, default=0.001, help='The Learning Rate.')
parser.add_argument('--momentum', type=float, default=0.9, help='Momentum.')
parser.add_argument('--decay', type=float, default=0., help='Weight decay (L2 penalty).')

# Checkpoints and Dynamics
parser.add_argument('--print_freq', default=200, type=int, metavar='N', help='print frequency (default: 200)')
parser.add_argument('--save_path', type=str, default='outputs/cifar_cnn', help='Folder to save logs')
parser.add_argument('--evaluate', dest='evaluate', action='store_true',default= False, help='evaluate model on validation set')
parser.add_argument('--dynamics', default=True, action='store_true', help='save training dynamics')
# Acceleration
parser.add_argument('--gpu', type=str, default=0)
parser.add_argument('--workers', type=int, default=2, help='number of data loading workers (default: 2)')
# random seed
parser.add_argument('--manualSeed', type=int, default='42', help='manual seed')

args = parser.parse_args()
args.use_cuda = True
args.device = f'cuda:{args.gpu}'

if args.manualSeed is None:
    args.manualSeed = random.randint(1, 10000)
random.seed(args.manualSeed)
torch.manual_seed(args.manualSeed)
if args.use_cuda:
    torch.cuda.manual_seed_all(args.manualSeed)
cudnn.benchmark = True

run_name = f"arch-{args.arch}_data-{args.dataset}_train_type-{args.train_type}_lr-{args.learning_rate}_bsz-{args.batch_size}_wd-{args.decay}"
wandb.init(project=f'matrix_completion_LoP_resnet_{args.dataset}', name=run_name, config=args)


def main():
    # Init logger
    print(args.save_path)
    args.save_path = os.path.join(args.save_path, args.dataset, f'{args.manualSeed}')
    log_path = os.path.join(args.save_path, 'log')
    if not os.path.isdir(log_path):
        os.makedirs(log_path)
    log = open(os.path.join(log_path, 'seed_{}_log.txt'.format(args.manualSeed)), 'w')
    print_log('save path : {}'.format(args.save_path), log)
    state = {k: v for k, v in args._get_kwargs()}
    print_log(state, log)
    print_log("Random Seed: {}".format(args.manualSeed), log)
    print_log("python version : {}".format(sys.version.replace('\n', ' ')), log)
    print_log("torch  version : {}".format(torch.__version__), log)
    print_log("cudnn  version : {}".format(torch.backends.cudnn.version()), log)
    print_log("Dataset: {}".format(args.dataset), log)
    print_log("Data Path: {}".format(args.data_path), log)
    print_log("Network: {}".format(args.arch), log)
    print_log("Batchsize: {}".format(args.batch_size), log)
    print_log("Learning Rate: {}".format(args.learning_rate), log)
    print_log("Momentum: {}".format(args.momentum), log)
    print_log("Weight Decay: {}".format(args.decay), log)

    # data loading
    train_data, _, test_loader = load_data(args)

    if args.dataset == 'cifar10':
        args.num_classes = 10
        args.num_samples = 50000
        args.num_iter = args.num_samples/args.batch_size
    if args.dataset == 'cifar100':
        args.num_classes = 100
        args.num_samples = 50000
        args.num_iter = args.num_samples/args.batch_size
    print_log("=> creating model '{}'".format(args.arch), log)
    # Init model, criterion, and optimizer
    if 'resnet' in args.arch:
        net = resnet.__dict__[args.arch](num_class = args.num_classes)

    elif 'vgg' in args.arch:
        net = vgg.VGG(args.arch, num_class = args.num_classes)

    elif args.arch == 'cnn':
        from models import cnn
        net = cnn.CNN(num_class = args.num_classes)

    print_log("=> network :\n {}".format(net), log)
    net = net.to(args.device)

    # define loss function (criterion) and optimizer
    criterion = torch.nn.CrossEntropyLoss().to(args.device)

    optimizer = torch.optim.SGD(net.parameters(), state['learning_rate'], momentum=state['momentum'],
                                weight_decay=state['decay'], nesterov=True)
    scheduler =  torch.optim.lr_scheduler.CosineAnnealingLR(optimizer = optimizer,
                                                        T_max =  args.epochs * args.num_iter)




    num_iters_per_chunk = len(train_data) // 2
    chunks = random_split(train_data, [num_iters_per_chunk] * 2)
    buffer = []

    # evaluation
    if args.evaluate:
        time1 = time.time()
        validate(test_loader, args, net, criterion, log) #
        time2 = time.time()
        print('function took %0.3f ms' % ((time2 - time1) * 1000.0))
        return

    # Main loop

    for chunk_idx in range(2):
        buffer.append(chunks[chunk_idx])
        _, chunk_loader, _ = load_data(args, buffer)


        start_time = time.time()
        epoch_time = AverageMeter()
        recorder = RecorderMeter(args.epochs)
        for epoch in range(args.epochs):
            if scheduler != None:
                current_learning_rate = scheduler.get_last_lr()[0]
            else:
                current_learning_rate = args.learning_rate
            need_hour, need_mins, need_secs = convert_secs2time(epoch_time.avg * (args.epochs - epoch))
            need_time = '[Need: {:02d}:{:02d}:{:02d}]'.format(need_hour, need_mins, need_secs)

            print_log(
                '\n==>>{:s} [Epoch={:03d}/{:03d}] {:s} [learning_rate={:6.4f}]'.format(time_string(), epoch, args.epochs,
                                                                                    need_time, args.learning_rate) \
                + ' [Best : Accuracy={:.2f}, Error={:.2f}]'.format(recorder.max_accuracy(False),
                                                                100 - recorder.max_accuracy(False)), log)

            # train for one epoch
            train_acc, train_loss = train(chunk_loader, args, net, criterion, optimizer, scheduler, epoch, log)

            # evaluate on validation set
            val_acc, val_loss = validate(test_loader, args, net, criterion, log)
            is_best = recorder.update(epoch, train_loss, train_acc, val_loss, val_acc)

            net.eval()
            avg_rank, layer_ranks = average_effective_rank(
                net,
                exclude_head=True,
                normalize=False
            )
            norm_avg_rank, norm_layer_ranks = average_effective_rank(
                net,
                exclude_head=True,
                normalize=True
            )
            print_log(f'  **Rank** AvgEffRank(norm=False) = {avg_rank:.4f}', log)
            print_log(f'  **Rank** AvgEffRank(norm=True) = {norm_avg_rank:.4f}', log)

            # NEW: wandb log
            # 기본 메트릭

            log_dict = {
                'epoch': epoch,
                'lr': current_learning_rate,
                'train/acc_top1': float(train_acc),
                'train/loss': float(train_loss),
                'val/acc_top1': float(val_acc),
                'val/loss': float(val_loss),
                'rank/avg_eff': float(avg_rank),
                'rank/norm_avg_eff': float(norm_avg_rank),
                'chunk_idx': chunk_idx+1
            }
            # 레이어별 상세(길이가 길 수 있으니 필요 시만)
            # for i, r in enumerate(layer_ranks):
            #     log_dict[f'layer_rank/layer_{i:02d}'] = float(r)
            #     log_dict[f'layer_rank/layer_{i:02d}'] = float(r)
            wandb.log(log_dict)


            # measure elapsed time
            epoch_time.update(time.time() - start_time)
            start_time = time.time()
            recorder.plot_curve(os.path.join(args.save_path, f'{args.manualSeed}_curve.png'))
            # if float(train_acc) >= 99.9:
                # break
        if args.train_type == 'cold':
            if 'resnet' in args.arch:
                net = resnet.__dict__[args.arch](num_class = args.num_classes)

            elif 'vgg' in args.arch:
                net = vgg.VGG(args.arch, num_class = args.num_classes)

            elif args.arch == 'cnn':
                from models import cnn
                net = cnn.CNN(num_class = args.num_classes)

            print_log("=> network :\n {}".format(net), log)
            net = net.to(args.device)

            # define loss function (criterion) and optimizer
            criterion = torch.nn.CrossEntropyLoss().to(args.device)

            optimizer = torch.optim.SGD(net.parameters(), state['learning_rate'], momentum=state['momentum'],
                                        weight_decay=state['decay'], nesterov=True)

    log.close()



# train function (forward, backward, update)
def train(train_loader, args, model, criterion, optimizer, scheduler, epoch, log):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    # switch to train mode
    model.train()
    end = time.time()

    for t, (input, target) in enumerate(train_loader):
        y = target.to(args.device)
        x = input.to(args.device)

        # compute output
        output = model(x)
        loss = criterion(output, y)


        # measure accuracy and record loss
        prec1, prec5 = accuracy(output.data, y, topk=(1, 5))
        losses.update(loss.item(), len(y))
        top1.update(prec1.item(), len(y))
        top5.update(prec5.item(), len(y))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()

        optimizer.step()
        if scheduler != None:
            scheduler.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if t % args.print_freq == 0:
            print_log('  Epoch: [{:03d}][{:03d}/{:03d}]   '
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})   '
                      'Data {data_time.val:.3f} ({data_time.avg:.3f})   '
                      'Loss {loss.val:.4f} ({loss.avg:.4f})   '
                      'Prec@1 {top1.val:.3f} ({top1.avg:.3f})   '
                      'Prec@5 {top5.val:.3f} ({top5.avg:.3f})   '.format(
                epoch, t, args.batch_size, batch_time=batch_time,
                data_time=data_time, loss=losses, top1=top1, top5=top5) + time_string(), log)


    print_log(
        '  **Train** Prec@1 {top1.avg:.3f} Prec@5 {top5.avg:.3f} Error@1 {error1:.3f}'.format(top1=top1, top5=top5,
                                                                                              error1=100 - top1.avg), log)
    return top1.avg, losses.avg


def validate(test_loader, args, model, criterion, log):
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to evaluate mode
    model.eval()
    with torch.no_grad():
        for i, (input, target) in enumerate(test_loader):
            y = target.to(args.device)
            x = input.to(args.device)

            # compute output
            output = model(x)
            loss = criterion(output, y)

            # measure accuracy and record loss
            prec1, prec5 = accuracy(output.data, y, topk=(1, 5))
            losses.update(loss.item(), len(y))
            top1.update(prec1.item(), len(y))
            top5.update(prec5.item(), len(y))

        print_log('  **Test** Prec@1 {top1.avg:.3f} Prec@5 {top5.avg:.3f} Error@1 {error1:.3f}'.format(top1=top1, top5=top5,
                                                                                                    error1=100 - top1.avg),
                log)

    return top1.avg, losses.avg


def print_log(print_string, log):
    print("{}".format(print_string))
    log.write('{}\n'.format(print_string))
    log.flush()


def save_checkpoint(state, is_best, save_path, filename):
    filename = os.path.join(save_path, filename)
    torch.save(state, filename)
    if is_best:
        bestname = os.path.join(save_path, f'{args.manualSeed}_best_ckpt.pth.tar')
        shutil.copyfile(filename, bestname)



def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].reshape(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


if __name__ == '__main__':
    main()
