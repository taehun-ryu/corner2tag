# Docker

Docker is optional. It provides the checkerboard executable and its native dependencies without installing them directly on the host.

## Requirements

- Docker Engine must be installed and accessible to the current user.
- The input must already be an HDF5 file in the [required layout](prerequisite.md#input-preconditions).
- Commands must be run from the repository root.

## Configure Calibration

### Parameters to Edit

Edit `docker/checkerboard.docker.yaml` and set the fields that describe your sensor, checkerboard, and temporal window:

```yaml
width: 346
height: 260

board_h: 5
board_w: 7
square_size: 0.35
fixed_window_us: 100000
```

`board_h` and `board_w` are inner-corner counts, not square counts. `fixed_window_us` must be a positive window length in microseconds.

### Docker I/O Paths — Do Not Edit

!!! warning "Do not edit these values"
    The Docker launcher requires the exact `h5_path` and `out_dir` values below.

```yaml
h5_path: "/data/input.h5"
out_dir: "results/checkerboard/"
```

The launcher maps the selected host file to `/data/input.h5` and writes results through the repository mount.

## Run

Pass the absolute path of the directory containing the HDF5 file, followed by the filename or its relative path inside that directory:

```bash
bash docker/run_checkerboard_docker.sh /absolute/path/to/data recording.h5
```

**Important:** Keep the space between `/absolute/path/to/data` and `recording.h5`. The launcher expects two separate arguments, not one full file path.

For example:

```bash
bash docker/run_checkerboard_docker.sh /home/user/datasets/events session01.h5
```

The second argument must not be an absolute path. A file in a nested directory is valid as long as it is relative to the first argument:

```bash
bash docker/run_checkerboard_docker.sh /home/user/datasets session01/events.h5
```

The launcher:

1. builds the `checkerboard` target from `docker/Dockerfile`;
2. mounts the repository at `/workspace`;
3. mounts the selected HDF5 file read-only at `/data/input.h5`;
4. runs checkerboard with `docker/checkerboard.docker.yaml`;
5. removes the container after the run.

The image is rebuilt when the command is run so that it reflects the current repository contents. Docker may reuse cached build layers.

## Outputs

Results are written on the host under:

```text
results/checkerboard/
└── run_YYYYMMDD_HHMMSS/
```

See [Checkerboard Outputs](checkerboard.md#outputs) for the generated files and how to inspect them.

## GUI

GUI output is disabled by default in Docker. Enable it with the environment variable below only when the host display environment is configured for container access:

```bash
CORNER2TAG_ENABLE_GUI=1 \
bash docker/run_checkerboard_docker.sh /absolute/path/to/data recording.h5
```

The launcher does not configure X11 or Wayland forwarding automatically. Saved output images are available regardless of GUI mode.

## Custom Image Tag

The default image tag is `corner2tag-checkerboard:latest`. Override it when needed:

```bash
CORNER2TAG_CHECKERBOARD_IMAGE_TAG=my-corner2tag:release2 \
bash docker/run_checkerboard_docker.sh /absolute/path/to/data recording.h5
```

## Troubleshooting

### `data directory not found`

The first argument must point to an existing directory. Prefer an absolute path.

### `expected input file not found`

The second argument must identify a file under the first directory. Check the filename and relative path.

### `h5_file_name must be a path relative to data directory`

Pass only the filename or a relative path as the second argument, not the full absolute path.

### Docker permission error

Confirm that Docker is running and that the current user is allowed to invoke `docker`. Follow the installation and post-installation instructions for the host operating system.
