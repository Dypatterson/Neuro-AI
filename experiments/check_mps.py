"""Check whether a Torch/MPS backend is available.

Run:
    python3 experiments/check_mps.py
"""

from __future__ import annotations


def main() -> None:
    try:
        import torch  # type: ignore
    except ModuleNotFoundError:
        print("torch: not installed")
        print("mps: unavailable")
        print("current kernel: pure Python reference backend")
        return

    print(f"torch: {torch.__version__}")
    print(f"mps: {torch.backends.mps.is_available()}")
    if torch.backends.mps.is_available():
        x = torch.randn(1024, device="mps")
        y = torch.randn(1024, device="mps")
        print(f"mps dot smoke test: {float((x * y).sum().cpu()):.4f}")
    try:
        from energy_memory.substrate.torch_fhrr import TorchFHRR
    except ModuleNotFoundError:
        return
    substrate = TorchFHRR(dim=256, seed=1)
    role = substrate.random_vector()
    filler = substrate.random_vector()
    recovered = substrate.unbind(substrate.bind(role, filler), role)
    print(f"torch fhrr device: {substrate.device}")
    print(f"torch fhrr round-trip: {substrate.similarity(filler, recovered):.4f}")


if __name__ == "__main__":
    main()
