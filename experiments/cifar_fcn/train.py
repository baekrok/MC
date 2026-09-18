import argparse
import csv
import json
from pathlib import Path
from typing import Dict, List

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms


class FCNet(nn.Module):
    """
    Fully connected network on flattened CIFAR10.

    input_dim -> h -> h -> ... -> h -> output_dim
    (depth hidden layers of width h)

    activation in {"linear", "relu", "tanh"}:
      - linear: no nonlinearity in hidden layers
      - relu: ReLU after each hidden layer
      - tanh: Tanh after each hidden layer
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        depth: int,
        init_type: str = "gaussian",
        activation: str = "linear",
        weight_std: float = 0.01,
        id_noise_std: float = 0.0,
    ):
        super().__init__()
        assert depth >= 1, "depth must be at least 1 (number of hidden layers)"
        assert activation in ["linear", "relu", "tanh"]
        assert init_type in ["gaussian", "identity"]

        self.depth = depth
        self.hidden_dim = hidden_dim
        self.init_type = init_type
        self.activation_name = activation
        self.weight_std = weight_std
        self.id_noise_std = id_noise_std

        layers = []

        # input layer: input_dim -> hidden_dim
        layers.append(nn.Linear(input_dim, hidden_dim, bias=True))

        # hidden h x h layers
        for _ in range(depth - 1):
            layers.append(nn.Linear(hidden_dim, hidden_dim, bias=True))

        # output layer: hidden_dim -> output_dim
        layers.append(nn.Linear(hidden_dim, output_dim, bias=True))

        self.layers = nn.ModuleList(layers)

        if activation == "linear":
            self.activation = None
        elif activation == "relu":
            self.activation = nn.ReLU()
        else:
            self.activation = nn.Tanh()

        self.reset_parameters()

    def reset_parameters(self):
        # first layer: input_dim -> h (always gaussian)
        first = self.layers[0]
        nn.init.normal_(first.weight, mean=0.0, std=self.weight_std)
        if first.bias is not None:
            nn.init.zeros_(first.bias)

        # middle hidden layers: h x h
        for idx in range(1, len(self.layers) - 1):
            layer = self.layers[idx]
            if self.init_type == "gaussian":
                nn.init.normal_(layer.weight, mean=0.0, std=self.weight_std)
            else:
                # identity init for h x h
                assert layer.weight.shape[0] == layer.weight.shape[1]
                h = layer.weight.shape[0]
                with torch.no_grad():
                    layer.weight.zero_()
                    layer.weight.add_(torch.eye(h))
                    if self.id_noise_std > 0:
                        layer.weight.add_(
                            torch.randn_like(layer.weight) * self.id_noise_std
                        )
            if layer.bias is not None:
                nn.init.zeros_(layer.bias)

        # last layer: h -> output_dim (always gaussian)
        last = self.layers[-1]
        nn.init.normal_(last.weight, mean=0.0, std=self.weight_std)
        if last.bias is not None:
            nn.init.zeros_(last.bias)

    def forward(self, x):
        # x: (batch, input_dim)
        h = x
        # first layer
        h = self.layers[0](h)
        # hidden layers with optional activation
        for idx in range(1, len(self.layers) - 1):
            if self.activation is None:
                h = self.layers[idx](h)
            else:
                h = self.activation(self.layers[idx](h))
        # last layer without activation
        h = self.layers[-1](h)
        return h

    @torch.no_grad()
    def effective_weight_product(self) -> torch.Tensor:
        """
        End to end weight W_eff for the linear product:
        W_eff = W_last * W_{L} * ... * W_1

        For non linear activations this is only an algebraic product,
        not the true mapping.
        """
        W_eff = self.layers[-1].weight  # (out, h)
        for idx in reversed(range(1, len(self.layers) - 1)):
            W_eff = W_eff @ self.layers[idx].weight
        W_eff = W_eff @ self.layers[0].weight
        return W_eff


def get_cifar10_loaders(
    batch_size: int, data_dir: str, num_workers: int = 4
):
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(
                mean=(0.4914, 0.4822, 0.4465),
                std=(0.2470, 0.2435, 0.2616),
            ),
            transforms.Lambda(lambda x: x.view(-1)),  # flatten 3x32x32 to 3072
        ]
    )

    train_set = torchvision.datasets.CIFAR10(
        root=data_dir, train=True, download=True, transform=transform
    )
    test_set = torchvision.datasets.CIFAR10(
        root=data_dir, train=False, download=True, transform=transform
    )

    train_loader = DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )
    test_loader = DataLoader(
        test_set,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )
    return train_loader, test_loader


@torch.no_grad()
def compute_matrix_metrics(W: torch.Tensor) -> Dict[str, float]:
    """
    Compute stable rank and effective rank of a matrix W.

    stable_rank = ||W||_F^2 / ||W||_2^2
    effective_rank = exp( H(p) ), where
      s: singular values, p_i = s_i^2 / sum_j s_j^2,
      H(p) = -sum p_i log p_i
    """
    # ensure 2D
    W2d = W
    s = torch.linalg.svdvals(W2d)
    if s.numel() == 0:
        return {"stable_rank": 0.0, "effective_rank": 0.0}

    s2 = s * s
    fro2 = s2.sum()
    if fro2 <= 0:
        return {"stable_rank": 0.0, "effective_rank": 0.0}

    # stable rank
    smax2 = s.max().item() ** 2
    if smax2 == 0:
        stable_rank = 0.0
    else:
        stable_rank = (fro2 / smax2).item()

    # effective rank
    p = s2 / fro2
    p_nonzero = p[p > 0]
    if p_nonzero.numel() == 0:
        effective_rank = 0.0
    else:
        entropy = -(p_nonzero * p_nonzero.log()).sum()
        effective_rank = torch.exp(entropy).item()

    return {
        "stable_rank": float(stable_rank),
        "effective_rank": float(effective_rank),
    }


def train_single_run(
    depth: int,
    hidden_dim: int,
    init_type: str,
    activation: str,
    optimizer_name: str,
    device: torch.device,
    epochs: int,
    batch_size: int,
    lr: float,
    weight_std: float,
    id_noise_std: float,
    seed: int,
    data_dir: str,
    num_workers: int,
) -> Dict:

    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    train_loader, test_loader = get_cifar10_loaders(
        batch_size=batch_size,
        data_dir=data_dir,
        num_workers=num_workers,
    )

    input_dim = 3 * 32 * 32
    num_classes = 10

    model = FCNet(
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        output_dim=num_classes,
        depth=depth,
        init_type=init_type,
        activation=activation,
        weight_std=weight_std,
        id_noise_std=id_noise_std,
    ).to(device)

    if optimizer_name == "adam":
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    elif optimizer_name == "sgd":
        optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9)
    else:
        raise ValueError(f"Unknown optimizer {optimizer_name}")

    criterion = nn.CrossEntropyLoss()

    epoch_stats: List[Dict] = []

    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        total_correct = 0
        total_samples = 0

        for images, targets in train_loader:
            images = images.to(device)
            targets = targets.to(device)

            optimizer.zero_grad()
            logits = model(images)
            loss = criterion(logits, targets)
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * images.size(0)
            preds = logits.argmax(dim=1)
            total_correct += (preds == targets).sum().item()
            total_samples += images.size(0)

        avg_loss = total_loss / total_samples
        train_acc = total_correct / total_samples

        model.eval()
        total_correct_test = 0
        total_samples_test = 0
        with torch.no_grad():
            for images, targets in test_loader:
                images = images.to(device)
                targets = targets.to(device)
                logits = model(images)
                preds = logits.argmax(dim=1)
                total_correct_test += (preds == targets).sum().item()
                total_samples_test += images.size(0)
        test_acc = total_correct_test / total_samples_test

        # rank metrics per epoch
        with torch.no_grad():
            W_eff = model.effective_weight_product().cpu()
            eff_metrics = compute_matrix_metrics(W_eff)

            layer_metrics = []
            for idx, layer in enumerate(model.layers):
                W = layer.weight.detach().cpu()
                m = compute_matrix_metrics(W)
                layer_metrics.append(
                    {
                        "layer_index": idx,
                        "shape": [int(W.shape[0]), int(W.shape[1])],
                        "stable_rank": m["stable_rank"],
                        "effective_rank": m["effective_rank"],
                    }
                )

        epoch_stats.append(
            {
                "epoch": epoch + 1,
                "train_loss": float(avg_loss),
                "train_acc": float(train_acc),
                "test_acc": float(test_acc),
                "effective_weight": {
                    "stable_rank": eff_metrics["stable_rank"],
                    "effective_rank": eff_metrics["effective_rank"],
                },
                "layers": layer_metrics,
            }
        )

        print(
            f"[opt={optimizer_name:4s}][init={init_type:8s}][act={activation:6s}] "
            f"depth {depth:2d}, epoch {epoch+1:3d}/{epochs:3d} "
            f"train_loss {avg_loss:.4f}, train_acc {train_acc:.4f}, "
            f"test_acc {test_acc:.4f}, "
            f"W_eff stable_rank {eff_metrics['stable_rank']:.2f}, "
            f"W_eff eff_rank {eff_metrics['effective_rank']:.2f}"
        )

    # 마지막 에폭 기준 summary (원하면 epoch_stats[-1] 기반)
    last_stats = epoch_stats[-1]
    last_eff = last_stats["effective_weight"]

    result = {
        "depth": depth,
        "hidden_dim": hidden_dim,
        "optimizer": optimizer_name,
        "init_type": init_type,
        "activation": activation,
        "seed": int(seed),
        "epochs": epochs,
        "batch_size": batch_size,
        "lr": float(lr),
        "weight_std": float(weight_std),
        "id_noise_std": float(id_noise_std),
        "final_train_acc": float(last_stats["train_acc"]),
        "final_test_acc": float(last_stats["test_acc"]),
        "final_W_eff_stable_rank": float(last_eff["stable_rank"]),
        "final_W_eff_effective_rank": float(last_eff["effective_rank"]),
        "epoch_stats": epoch_stats,
    }
    return result


def parse_args():
    parser = argparse.ArgumentParser(
        description="Single CIFAR10 FC rank experiment with per epoch metrics"
    )
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--depth", type=int, required=True)
    parser.add_argument("--hidden-dim", type=int, default=512)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--data-dir", type=str, default="data")
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument(
        "--optimizer",
        type=str,
        default="adam",
        choices=["adam", "sgd"],
    )
    parser.add_argument(
        "--init-type",
        type=str,
        default="gaussian",
        choices=["gaussian", "identity"],
    )
    parser.add_argument(
        "--activation",
        type=str,
        default="linear",
        choices=["linear", "relu", "tanh"],
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--weight-std", type=float, default=0.01)
    parser.add_argument("--id-noise-std", type=float, default=0.0)
    parser.add_argument(
        "--out-json",
        type=str,
        required=True,
        help="output json path for this run",
    )
    parser.add_argument(
        "--out-csv",
        type=str,
        required=True,
        help="output csv path for this run (summary row)",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    if torch.cuda.is_available():
        device = torch.device(f"cuda:{args.gpu}")
    else:
        device = torch.device("cpu")

    print("Using device:", device)

    result = train_single_run(
        depth=args.depth,
        hidden_dim=args.hidden_dim,
        init_type=args.init_type,
        activation=args.activation,
        optimizer_name=args.optimizer,
        device=device,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        weight_std=args.weight_std,
        id_noise_std=args.id_noise_std,
        seed=args.seed,
        data_dir=args.data_dir,
        num_workers=args.num_workers,
    )

    # save json (full epoch_stats 포함)
    Path(args.out_json).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out_csv).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out_json, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)
    print(f"Saved json to {args.out_json}")

    # save csv summary row
    row = {
        k: v
        for k, v in result.items()
        if k != "epoch_stats"
    }
    # 필요하면 epoch_stats도 문자열로 넣을 수 있음
    # row["epoch_stats"] = json.dumps(result["epoch_stats"])

    keys = sorted(row.keys())
    write_header = True
    try:
        with open(args.out_csv, "r", encoding="utf-8"):
            write_header = False
    except FileNotFoundError:
        write_header = True

    with open(args.out_csv, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        if write_header:
            writer.writeheader()
        writer.writerow(row)

    print(f"Appended csv row to {args.out_csv}")


if __name__ == "__main__":
    main()
