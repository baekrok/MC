"""Deep matrix-factorization experiment for low-rank matrix completion."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from torch import nn


def effective_rank(matrix: np.ndarray, eps: float = 1e-8) -> float:
    """Return the entropy-based effective rank of a matrix."""
    singular_values = np.linalg.svd(matrix, compute_uv=False)
    total = singular_values.sum()
    if total <= eps:
        return 0.0
    probabilities = singular_values / total
    entropy = -(probabilities * np.log(probabilities + eps)).sum()
    return float(np.exp(entropy))


class DeepMatrixFactorization(nn.Module):
    """Represent a square matrix as the product of ``depth`` factors."""

    def __init__(
        self,
        dimension: int,
        depth: int,
        init_scale: float,
        init_type: str,
        mimic_denominator: float,
        rng: np.random.Generator,
    ) -> None:
        super().__init__()
        if init_type == "gaussian_mimic":
            initial = torch.full(
                (dimension, dimension),
                init_scale / mimic_denominator,
                dtype=torch.float32,
            )
            initial.fill_diagonal_(init_scale)
        elif init_type == "gaussian":
            initial = torch.tensor(
                rng.normal(scale=init_scale, size=(dimension, dimension)),
                dtype=torch.float32,
            )
        else:
            raise ValueError(f"Unknown initialization type: {init_type}")

        # Keep the original experiments' tied initialization: every factor starts
        # from the same sampled matrix, then trains as an independent parameter.
        self.factors = nn.ParameterList(
            nn.Parameter(initial.clone()) for _ in range(depth)
        )

    def forward(self) -> torch.Tensor:
        product = self.factors[0]
        for factor in self.factors[1:]:
            product = product @ factor
        return product


def run_trial(args: argparse.Namespace, alpha: float, seed: int) -> bool:
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)

    left = torch.randn(args.dimension, args.rank)
    right = torch.randn(args.dimension, args.rank)
    ground_truth = (left @ right.T).to(args.device)
    mask = (
        torch.rand(args.dimension, args.dimension) < args.mask_probability
    ).float().to(args.device)
    observed = mask * ground_truth

    init_scale = alpha ** (1 / args.depth) / np.sqrt(args.dimension)
    model = DeepMatrixFactorization(
        dimension=args.dimension,
        depth=args.depth,
        init_scale=init_scale,
        init_type=args.init_type,
        mimic_denominator=args.mimic_denominator,
        rng=rng,
    ).to(args.device)
    optimizer = torch.optim.SGD(model.parameters(), lr=args.learning_rate)

    for epoch in range(1, args.max_epochs + 1):
        optimizer.zero_grad()
        prediction = model()
        loss = ((mask * prediction - observed) ** 2).mean()
        loss.backward()
        optimizer.step()

        if epoch % args.print_every == 0:
            rank = effective_rank(prediction.detach().cpu().numpy())
            print(
                f"alpha={alpha:.0e} seed={seed} epoch={epoch} "
                f"observed_mse={loss.item():.6g} effective_rank={rank:.5f}"
            )

        if loss.item() < args.threshold:
            prediction_np = prediction.detach().cpu().numpy()
            payload = {
                "alpha": alpha,
                "depth": args.depth,
                "seed": seed,
                "epochs": epoch,
                "observed_mse": float(loss.item()),
                "effective_rank": effective_rank(prediction_np),
                "singular_values": np.linalg.svd(
                    prediction_np, compute_uv=False
                ),
            }
            output_path = (
                args.output_dir
                / f"depth_{args.depth}"
                / f"alpha_{alpha:.0e}_seed_{seed}.npy"
            )
            output_path.parent.mkdir(parents=True, exist_ok=True)
            np.save(output_path, payload)
            print(f"Saved {output_path}")
            return True

    print(
        f"Did not converge: alpha={alpha:.0e}, seed={seed}, "
        f"loss={loss.item():.6g}"
    )
    return False


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a deep linear factorization on a masked low-rank matrix."
    )
    parser.add_argument("--depth", type=int, default=2)
    parser.add_argument("--dimension", type=int, default=100)
    parser.add_argument("--rank", type=int, default=5)
    parser.add_argument("--mask-probability", type=float, default=0.2)
    parser.add_argument("--alphas", type=float, nargs="+", default=[1e-2, 1e-4])
    parser.add_argument("--seed-start", type=int, default=80)
    parser.add_argument("--seed-end", type=int, default=100)
    parser.add_argument("--threshold", type=float, default=1e-5)
    parser.add_argument("--max-epochs", type=int, default=10_000_000)
    parser.add_argument("--print-every", type=int, default=10_000)
    parser.add_argument("--learning-rate", type=float, default=0.2)
    parser.add_argument(
        "--init-type",
        choices=["gaussian", "gaussian_mimic"],
        default="gaussian",
    )
    parser.add_argument("--mimic-denominator", type=float, default=100.0)
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    parser.add_argument(
        "--output-dir", type=Path, default=Path("outputs/matrix_completion")
    )
    args = parser.parse_args()

    if args.depth < 1:
        parser.error("--depth must be at least 1")
    if args.seed_end <= args.seed_start:
        parser.error("--seed-end must be greater than --seed-start")
    if not 0 < args.mask_probability <= 1:
        parser.error("--mask-probability must be in (0, 1]")
    return args


def main() -> None:
    args = parse_args()
    args.device = torch.device(args.device)
    for alpha in args.alphas:
        for seed in range(args.seed_start, args.seed_end):
            run_trial(args, alpha, seed)


if __name__ == "__main__":
    main()
