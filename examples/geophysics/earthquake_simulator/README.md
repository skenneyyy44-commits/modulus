# 3D Earthquake Simulator (Acoustic Wave Equation)

This example provides a lightweight, educational 3D earthquake simulator that
solves the scalar acoustic wave equation on a uniform grid. It uses a Ricker
wavelet as the source and writes wavefield snapshots to a compressed `.npz`
file.

## Quick start

```bash
python simulate.py --output outputs/earthquake_run.npz
```

The output file contains:

- `snapshots`: wavefield snapshots sampled every `--snapshot-interval` steps.
- `metadata`: grid spacing, time step, and wave speed.
- `source_location`: the source index in the grid.

## Configuration

Common flags:

- `--nx`, `--ny`, `--nz`: grid dimensions (default: 64 × 64 × 64)
- `--dx`: grid spacing in meters (default: 10.0)
- `--dt`: time step in seconds (default: 0.001)
- `--nt`: number of time steps (default: 400)
- `--velocity`: homogeneous wave speed (default: 2500.0 m/s)
- `--source-frequency`: Ricker wavelet peak frequency in Hz (default: 12.0)
- `--snapshot-interval`: snapshot cadence (default: 20)
- `--source-location`: explicit source indices, e.g. `--source-location 32 32 32`

## Notes

- This is a simple acoustic model intended for demonstration and education.
- The solver uses a second-order finite-difference time domain (FDTD) scheme and
  zero-value boundary conditions.
