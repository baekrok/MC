import argparse
import json
import random
from copy import deepcopy
from pathlib import Path
from typing import List, Dict, Tuple

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from tqdm import tqdm


# ----------------------------
# Utils
# ----------------------------
def set_seed(seed: int = 1337):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(False)


def accuracy(model: nn.Module, loader: DataLoader, device: torch.device) -> float:
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            pred = logits.argmax(dim=1)
            correct += (pred == y).sum().item()
            total += y.numel()
    return 100.0 * correct / total


def evaluate_all(model: nn.Module, loaders: Dict[str, DataLoader], device: torch.device) -> Dict[str, float]:
    return {split: accuracy(model, ld, device) for split, ld in loaders.items()}
@torch.no_grad()
def effective_rank_from_svals(svals: torch.Tensor, eps: float = 1e-12) -> float:
    """
    Entropy-based effective rank: r_eff = exp(H(p)),  p_i = s_i^2 / sum_j s_j^2
    """
    p = svals**2
    denom = p.sum()
    if denom <= eps:
        return 0.0
    p = p / denom
    H = -(p * torch.log(p + eps)).sum()
    return float(torch.exp(H))

@torch.no_grad()
def layer_effective_rank(W: torch.Tensor) -> float:
    """
    W: [out, in]; compute singular values on CPU for stability.
    """
    W_cpu = W.detach().float().cpu()
    # full_matrices=False -> min(out,in) singular values
    s = torch.linalg.svd(W_cpu, full_matrices=False).S
    return effective_rank_from_svals(s)

@torch.no_grad()
def model_effective_rank(
    model: nn.Module,
    exclude_head: bool = True,
    return_per_layer: bool = False,
) -> Tuple[float, List[float]]:
    """
    (헤드 제외) 모든 Linear의 effective rank를 계산해서 평균을 반환.
    """
    ranks = []
    for _, lin in model.linear_layers(exclude_head=exclude_head):
        ranks.append(layer_effective_rank(lin.weight.data))
    avg_rank = float(sum(ranks) / max(1, len(ranks)))
    if return_per_layer:
        return avg_rank, ranks
    return avg_rank, []

# ----------------------------
# Model: L-layer MLP for CIFAR-10
# ----------------------------
class MLP(nn.Module):
    def __init__(self, layers: int = 5, hidden: int = 1024, num_classes: int = 10, dropout: float = 0.0):
        """Simple MLP: [Flatten] -> Linear/ReLU x (layers-1) -> Linear(num_classes)."""
        super().__init__()
        assert layers >= 2, "layers must be >= 2 (input->...->output)"
        in_dim = 3 * 32 * 32
        modules: List[nn.Module] = [nn.Flatten()]
        dims = [in_dim] + [hidden] * (layers - 2) + [num_classes]
        for i in range(len(dims) - 1):
            modules.append(nn.Linear(dims[i], dims[i + 1], bias=True))
            if i < len(dims) - 2:
                modules.append(nn.ReLU(inplace=True))
                if dropout > 0:
                    modules.append(nn.Dropout(dropout))
        self.net = nn.Sequential(*modules)

    def forward(self, x):
        return self.net(x)

    def linear_layers(self, exclude_head: bool = False):
        """(idx, Linear) 목록. exclude_head=True면 마지막 Linear(클래스 헤드)를 제외."""
        idx_and_layers = []
        for idx, m in enumerate(self.net):
            if isinstance(m, nn.Linear):
                idx_and_layers.append((idx, m))
        if exclude_head and len(idx_and_layers) > 0:
            idx_and_layers = idx_and_layers[:-1]  # 마지막 Linear 제외
        return idx_and_layers

# ----------------------------
# Low-rank approximation
# ----------------------------
@torch.no_grad()
def truncated_svd_weight(W: torch.Tensor, rank: int) -> torch.Tensor:
    """
    Return low-rank approximation of W using top-k singular values.
    W: [out, in]
    """
    # Compute full SVD on CPU for numerical stability if sizes are moderate
    device = W.device
    W_cpu = W.detach().float().cpu()
    U, S, Vh = torch.linalg.svd(W_cpu, full_matrices=False)
    k = max(1, min(rank, S.shape[0]))
    Uk = U[:, :k]
    Sk = S[:k]
    Vhk = Vh[:k, :]
    Wk = (Uk * Sk) @ Vhk  # (out,k) * (k,in) -> (out,in)
    return Wk.to(device).type_as(W)


@torch.no_grad()
def apply_lowrank_to_layer(linear: nn.Linear, rank_ratio: float):
    W = linear.weight.data
    out_dim, in_dim = W.shape
    max_rank = min(out_dim, in_dim)
    rank = max(1, int(round(rank_ratio * max_rank)))
    W_lr = truncated_svd_weight(W, rank)
    linear.weight.copy_(W_lr)
    # Bias는 그대로 유지 (W 저랭크 치환만)
    return rank


def clone_and_lowrank_one_layer(model: nn.Module, layer_idx: int, rank_ratio: float, exclude_head: bool = True):
    model2 = deepcopy(model)
    linear_layers = [m for (_, m) in model2.linear_layers(exclude_head=exclude_head)]
    assert 0 <= layer_idx < len(linear_layers), "Invalid layer index"
    chosen = linear_layers[layer_idx]
    r = apply_lowrank_to_layer(chosen, rank_ratio)
    return model2, r

def clone_and_lowrank_prefix(model: nn.Module, upto_idx: int, rank_ratio: float, exclude_head: bool = True):
    model2 = deepcopy(model)
    layers = [m for (_, m) in model2.linear_layers(exclude_head=exclude_head)]
    ranks = []
    for i in range(upto_idx + 1):
        r = apply_lowrank_to_layer(layers[i], rank_ratio)
        ranks.append(r)
    return model2, ranks


# ----------------------------
# Training loop
# ----------------------------
def train(model: nn.Module,
          train_loader: DataLoader,
          test_loader: DataLoader,
          device: torch.device,
          epochs: int = 20,
          lr: float = 1e-3,
          weight_decay: float = 0.0,
          use_amp: bool = True) -> Dict[str, float]:
    model.to(device)
    opt = optim.SGD(model.parameters(), lr=lr, weight_decay=weight_decay, momentum=0.9)
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp and device.type == "cuda")
    criterion = nn.CrossEntropyLoss()

    # NEW: rank 기록용
    rank_history = {
        "avg_eff_rank": [],          # 매 에폭 평균
        "per_layer_eff_rank": []     # 매 에폭 레이어별 리스트
    }

    best = {"epoch": -1, "test_acc": 0.0}
    for ep in range(1, epochs + 1):
        model.train()
        pbar = tqdm(train_loader, desc=f"Epoch {ep}/{epochs}", leave=False)
        for x, y in pbar:
            x, y = x.to(device), y.to(device)
            opt.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=use_amp and device.type == "cuda"):
                logits = model(x)
                loss = criterion(logits, y)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            pbar.set_postfix(loss=float(loss))

        # NEW: 에폭 종료 시 effective rank 측정 (헤드 제외)
        model.eval()
        avg_rank, per_layer = model_effective_rank(model, exclude_head=True, return_per_layer=True)
        rank_history["avg_eff_rank"].append(avg_rank)
        rank_history["per_layer_eff_rank"].append(per_layer)
        print(f"[Epoch {ep}] avg effective rank (excl. head): {avg_rank:.3f}  per-layer: {[round(r,3) for r in per_layer]}")

        # 기존 정확도 측정은 유지(원하면 꺼도 됨)
        test_acc = accuracy(model, test_loader, device)
        if test_acc > best["test_acc"]:
            best = {"epoch": ep, "test_acc": test_acc}
        print(best)

    # NEW: history를 반환 객체에 넣기
    best["rank_history"] = rank_history
    return best


# ----------------------------
# Data
# ----------------------------
def make_loaders(
    batch_size: int = 128,
    num_workers: int = 4,
    data_dir: str = "data",
) -> Tuple[DataLoader, DataLoader]:
    train_tf = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465),
                             (0.2470, 0.2435, 0.2616)),
    ])
    test_tf = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465),
                             (0.2470, 0.2435, 0.2616)),
    ])
    train_ds = datasets.CIFAR10(root=data_dir, train=True, download=True, transform=train_tf)
    test_ds = datasets.CIFAR10(root=data_dir, train=False, download=True, transform=test_tf)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=512, shuffle=False,
                             num_workers=num_workers, pin_memory=True)
    return train_loader, test_loader


# ----------------------------
# Orchestrator
# ----------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--layers", type=int, default=5, help="Total MLP layers including output (>=2)")
    ap.add_argument("--hidden", type=int, default=1024)
    ap.add_argument("--dropout", type=float, default=0.0)
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight-decay", type=float, default=0.0)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--num-workers", type=int, default=4)
    ap.add_argument("--data-dir", type=str, default="data")

    # LOW-RANK 관련 인자/로직은 더 이상 사용하지 않으므로 제거해도 되지만,
    # 기존 실험 재현성을 위해 남겨두되, 아래에서는 사용하지 않음.

    # NEW: 결과 경로 (랭크 로그 & 플롯)
    ap.add_argument("--rank-log", type=str, default="effective_rank_log.json")
    ap.add_argument("--rank-plot", type=str, default="effective_rank_plot.png")

    args = ap.parse_args()
    set_seed(args.seed)

    device = torch.device(args.device)
    train_loader, test_loader = make_loaders(
        args.batch_size, args.num_workers, args.data_dir
    )

    model = MLP(layers=args.layers, hidden=args.hidden, dropout=args.dropout).to(device)
    print(model)

    print("\n==> Training & Tracking Effective Rank ...")
    best = train(model, train_loader, test_loader, device,
                 epochs=args.epochs, lr=args.lr, weight_decay=args.weight_decay, use_amp=True)

    # NEW: 로그 저장
    rank_hist = best.get("rank_history", {})
    payload = {
        "config": vars(args),
        "best": {"epoch": best["epoch"], "test_acc": best["test_acc"]},
        "num_linear_excl_head": len(model.linear_layers(exclude_head=True)),
        "linear_indices_excl_head": [idx for (idx, _) in model.linear_layers(exclude_head=True)],
        "rank_history": rank_hist,
    }
    Path(args.rank_log).parent.mkdir(parents=True, exist_ok=True)
    Path(args.rank_plot).parent.mkdir(parents=True, exist_ok=True)
    with open(args.rank_log, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    print(f"[Saved] {args.rank_log}")

    # NEW: 플롯 저장
    try:
        import matplotlib.pyplot as plt
        avg = rank_hist.get("avg_eff_rank", [])
        xs = list(range(1, len(avg) + 1))
        plt.figure()
        plt.plot(xs, avg, marker="o")
        plt.xlabel("Epoch")
        plt.ylabel("Avg Effective Rank (excluding head)")
        plt.title("Epoch-wise Average Effective Rank Across Linear Layers")
        plt.grid(True, which="both", linestyle="--", linewidth=0.5)
        plt.tight_layout()
        plt.savefig(args.rank_plot, dpi=200)
        print(f"[Saved] {args.rank_plot}")
    except Exception as e:
        print(f"[Plot skipped] {e}")

if __name__ == "__main__":
    main()
