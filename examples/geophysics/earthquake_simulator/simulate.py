"""Simple 3D acoustic wave equation earthquake simulator.

This script solves a scalar 3D wave equation on a uniform grid with a Ricker
wavelet source. It is intended as a lightweight educational example rather than
an industry-grade solver.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass
class SimulationConfig:
    nx: int
    ny: int
    nz: int
    dx: float
    dt: float
    nt: int
    velocity: float
    source_frequency: float
    source_location: tuple[int, int, int]
    snapshot_interval: int
    output_path: Path | None


def ricker_wavelet(t: np.ndarray, frequency: float) -> np.ndarray:
    """Return a Ricker wavelet evaluated at times ``t``."""
    pi_f_t = np.pi * frequency * t
    return (1.0 - 2.0 * pi_f_t**2) * np.exp(-pi_f_t**2)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="3D acoustic wave equation earthquake simulator",
    )
    parser.add_argument("--nx", type=int, default=64, help="Grid points in x")
    parser.add_argument("--ny", type=int, default=64, help="Grid points in y")
    parser.add_argument("--nz", type=int, default=64, help="Grid points in z")
    parser.add_argument("--dx", type=float, default=10.0, help="Grid spacing (m)")
    parser.add_argument("--dt", type=float, default=0.001, help="Time step (s)")
    parser.add_argument("--nt", type=int, default=400, help="Number of time steps")
    parser.add_argument(
        "--velocity",
        type=float,
        default=2500.0,
        help="Homogeneous wave speed (m/s)",
    )
    parser.add_argument(
        "--source-frequency",
        type=float,
        default=12.0,
        help="Ricker wavelet peak frequency (Hz)",
    )
    parser.add_argument(
        "--source-location",
        type=int,
        nargs=3,
        default=None,
        metavar=("SX", "SY", "SZ"),
        help="Source indices (x y z). Defaults to grid center.",
    )
    parser.add_argument(
        "--snapshot-interval",
        type=int,
        default=20,
        help="Interval (timesteps) between saved wavefield snapshots",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional path to save npz output (snapshots + metadata)",
    )
    return parser


def parse_config(args: argparse.Namespace) -> SimulationConfig:
    if args.source_location is None:
        source_location = (args.nx // 2, args.ny // 2, args.nz // 2)
    else:
        source_location = tuple(args.source_location)

    return SimulationConfig(
        nx=args.nx,
        ny=args.ny,
        nz=args.nz,
        dx=args.dx,
        dt=args.dt,
        nt=args.nt,
        velocity=args.velocity,
        source_frequency=args.source_frequency,
        source_location=source_location,
        snapshot_interval=args.snapshot_interval,
        output_path=args.output,
    )


def simulate(config: SimulationConfig) -> tuple[np.ndarray, dict[str, float]]:
    """Run the wave simulation and return snapshots with metadata."""
    nx, ny, nz = config.nx, config.ny, config.nz
    u_prev = np.zeros((nx, ny, nz), dtype=np.float32)
    u_curr = np.zeros_like(u_prev)
    u_next = np.zeros_like(u_prev)

    times = np.arange(config.nt) * config.dt
    source_signal = ricker_wavelet(times, config.source_frequency).astype(np.float32)

    snapshots: list[np.ndarray] = []

    c = config.velocity
    dt_dx = (c * config.dt / config.dx) ** 2

    sx, sy, sz = config.source_location

    for step in range(config.nt):
        laplacian = (
            -6.0 * u_curr
            + np.roll(u_curr, 1, axis=0)
            + np.roll(u_curr, -1, axis=0)
            + np.roll(u_curr, 1, axis=1)
            + np.roll(u_curr, -1, axis=1)
            + np.roll(u_curr, 1, axis=2)
            + np.roll(u_curr, -1, axis=2)
        )

        u_next = 2.0 * u_curr - u_prev + dt_dx * laplacian
        u_next[sx, sy, sz] += source_signal[step]

        if step % config.snapshot_interval == 0:
            snapshots.append(u_next.copy())

        u_prev, u_curr, u_next = u_curr, u_next, u_prev

    metadata = {
        "dx": config.dx,
        "dt": config.dt,
        "velocity": config.velocity,
        "source_frequency": config.source_frequency,
    }
    return np.stack(snapshots, axis=0), metadata


def save_output(
    output_path: Path,
    snapshots: np.ndarray,
    metadata: dict[str, float],
    config: SimulationConfig,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_path,
        snapshots=snapshots,
        metadata=metadata,
        source_location=np.array(config.source_location),
        snapshot_interval=config.snapshot_interval,
    )


def main() -> None:
    parser = build_parser()
    config = parse_config(parser.parse_args())

    snapshots, metadata = simulate(config)

    if config.output_path is not None:
        save_output(config.output_path, snapshots, metadata, config)
        print(f"Saved {snapshots.shape[0]} snapshots to {config.output_path}")
    else:
        print(
            "Simulation complete. Provide --output to save snapshots; "
            f"generated {snapshots.shape[0]} snapshots in-memory."
        )


if __name__ == "__main__":
    main()
