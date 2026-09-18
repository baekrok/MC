import time
import numpy as np
import matplotlib.pyplot as plt
import random
import pickle
import torch

# === add below helpers (e.g., after print_log/save_checkpoint) ===
# NEW: effective rank utilities
@torch.no_grad()
def _effective_rank_from_svals(svals: torch.Tensor, eps: float = 1e-12) -> float:
    """Entropy-based effective rank: r_eff = exp(H(p)),  p_i = s_i^2 / sum_j s_j^2"""
    if svals.numel() == 0:
        return 0.0
    p = (svals ** 2).float()
    denom = p.sum()
    if denom <= eps:
        return 0.0
    p = p / denom
    H = -(p * torch.log(p + eps)).sum()
    return float(torch.exp(H))

@torch.no_grad()
def _layer_effective_rank_from_weight(W: torch.Tensor) -> float:
    """
    W: (out_dim, in_dim) 2D weight matrix on any device.
    SVD는 CPU float로 계산(안정성).
    """
    s = torch.linalg.svdvals(W.detach().float().cpu())  # returns singular values only
    return _effective_rank_from_svals(s)

@torch.no_grad()
def average_effective_rank(model: torch.nn.Module, exclude_head: bool = True, normalize: bool = False):
    """
    모델의 Conv2d/Linear 가중치에 대해 레이어별 effective rank를 구해 평균 반환.
    - exclude_head: 마지막 nn.Linear(클래스 헤드) 제외
    - normalize: 각 레이어 r_eff / min(out_dim, in_dim)로 정규화(옵션)
    """
    mats = []
    last_fc = None
    for m in model.modules():
        if isinstance(m, torch.nn.Linear):
            W = m.weight.data
            mats.append(("linear", W))
            last_fc = ("linear", W)  # 마지막으로 본 linear를 저장
        elif isinstance(m, torch.nn.Conv2d):
            W = m.weight.data  # [C_out, C_in, kH, kW]
            C_out, C_in, kH, kW = W.shape
            W2D = W.reshape(C_out, C_in * kH * kW)
            mats.append(("conv", W2D))

    if exclude_head and last_fc is not None:
        # 마지막 Linear를 제거
        # (mats에서 Linear들 중 마지막 항목을 pop)
        for i in range(len(mats) - 1, -1, -1):
            if mats[i][0] == "linear":
                mats.pop(i)
                break

    ranks = []
    for kind, W in mats:
        r = _layer_effective_rank_from_weight(W)
        if normalize:
            out_dim, in_dim = W.shape
            r = r / float(min(out_dim, in_dim) if min(out_dim, in_dim) > 0 else 1.0)
        ranks.append(r)

    avg_rank = float(np.mean(ranks)) if len(ranks) > 0 else 0.0
    return avg_rank, ranks


class AverageMeter(object):
  """Computes and stores the average and current value"""
  def __init__(self):
    self.reset()

  def reset(self):
    self.val = 0
    self.avg = 0
    self.sum = 0
    self.count = 0

  def update(self, val, n=1):
    self.val = val
    self.sum += val * n
    self.count += n
    self.avg = self.sum / self.count


class RecorderMeter(object):
  """Computes and stores the minimum loss value and its epoch index"""
  def __init__(self, total_epoch):
    self.reset(total_epoch)

  def reset(self, total_epoch):
    assert total_epoch > 0
    self.total_epoch   = total_epoch
    self.current_epoch = 0
    self.epoch_losses  = np.zeros((self.total_epoch, 2), dtype=np.float64) # [epoch, train/val]
    self.epoch_losses  = self.epoch_losses - 1

    self.epoch_accuracy= np.zeros((self.total_epoch, 2), dtype=np.float64) # [epoch, train/val]
    self.epoch_accuracy= self.epoch_accuracy

  def update(self, idx, train_loss, train_acc, val_loss, val_acc):
    assert idx >= 0 and idx < self.total_epoch, 'total_epoch : {} , but update with the {} index'.format(self.total_epoch, idx)
    self.epoch_losses  [idx, 0] = train_loss
    self.epoch_losses  [idx, 1] = val_loss
    self.epoch_accuracy[idx, 0] = train_acc
    self.epoch_accuracy[idx, 1] = val_acc
    self.current_epoch = idx + 1
    return self.max_accuracy(False) == val_acc

  def max_accuracy(self, istrain):
    if self.current_epoch <= 0: return 0
    if istrain: return self.epoch_accuracy[:self.current_epoch, 0].max()
    else:       return self.epoch_accuracy[:self.current_epoch, 1].max()

  def plot_curve(self, save_path):
    title = 'the accuracy/loss curve of train/val'
    dpi = 150
    width, height = 1200, 800
    legend_fontsize = 13
    scale_distance = 48.8
    figsize = width / float(dpi), height / float(dpi)

    fig = plt.figure(figsize=figsize)
    x_axis = np.array([i for i in range(self.total_epoch)]) # epochs
    y_axis = np.zeros(self.total_epoch)

    plt.xlim(0, self.total_epoch)
    plt.ylim(0, 100)
    interval_y = 5
    interval_x = 20
    plt.xticks(np.arange(0, self.total_epoch + interval_x, interval_x))
    plt.yticks(np.arange(0, 100 + interval_y, interval_y))
    plt.grid()
    plt.title(title, fontsize=20)
    plt.xlabel('the training epoch', fontsize=16)
    plt.ylabel('accuracy', fontsize=16)

    y_axis[:] = self.epoch_accuracy[:, 0]
    plt.plot(x_axis, y_axis, color='b', linestyle='-', label='train-accuracy', lw=2)
    plt.legend(loc=4, fontsize=legend_fontsize)

    y_axis[:] = self.epoch_accuracy[:, 1]
    plt.plot(x_axis, y_axis, color='r', linestyle='-', label='valid-accuracy', lw=2)
    plt.legend(loc=4, fontsize=legend_fontsize)


    y_axis[:] = self.epoch_losses[:, 0]
    plt.plot(x_axis, y_axis*50, color='b', linestyle=':', label='train-loss-x50', lw=2)
    plt.legend(loc=4, fontsize=legend_fontsize)


    if save_path is not None:
      fig.savefig(save_path, dpi=dpi, bbox_inches='tight')
      print ('---- save figure {} into {}'.format(title, save_path))
    plt.close(fig)


def time_string():
  ISOTIMEFORMAT='%Y-%m-%d %X'
  string = '[{}]'.format(time.strftime( ISOTIMEFORMAT, time.gmtime(time.time()+28800) ))
  return string

def convert_secs2time(epoch_time):
  need_hour = int(epoch_time / 3600)
  need_mins = int((epoch_time - 3600*need_hour) / 60)
  need_secs = int(epoch_time - 3600*need_hour - 60*need_mins)
  return need_hour, need_mins, need_secs

def time_file_str():
  ISOTIMEFORMAT='%Y-%m-%d'
  string = '{}'.format(time.strftime( ISOTIMEFORMAT, time.gmtime(time.time()) ))
  return string + '-{}'.format(random.randint(1, 10000))

def timing(f):
    def wrap(*args):
        time1 = time.time()
        ret = f(*args)
        time2 = time.time()
        print ('%s function took %0.3f ms' % (f.__name__, (time2-time1)*1000.0))
        return ret
    return wrap
