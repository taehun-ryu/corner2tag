# Prerequisites

This page defines the baseline setup required before running the public release.

## Tested Environment

The following environment has been tested:

- Ubuntu 22.04
- OpenCV 4.5.4
- C++17 compiler (GCC/Clang)
- CMake 3.16+

Newer Ubuntu and OpenCV versions may work, but are not verified.

**Note:** If you have trouble with environment setup, you can optionally use [Docker](docker.md).

## Required Dependencies

Install core packages:

```bash
sudo apt-get update
sudo apt-get install -y build-essential cmake pkg-config libopencv-dev libhdf5-dev libeigen3-dev
```

Install Ceres Solver (2.2.0 recommended) using the official guide:

- <http://ceres-solver.org/installation.html>

## Input Preconditions

Input HDF5 must include the following dataset layout:

```text
/
└── events
    ├── ts   (uint64, microseconds)
    ├── xs   (uint16)
    ├── ys   (uint16)
    └── ps   (uint8, 0 or 1)
```

- `ts`, `xs`, `ys`, and `ps` must be one-dimensional arrays of equal length.
- `ts` must be integer microseconds (`uint64`).
