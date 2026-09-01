# Quick Start

corner2tag accepts HDF5 event files and can run natively or in Docker.

## 1. Prepare an HDF5 File

The input must contain `events/ts`, `events/xs`, `events/ys`, and `events/ps`. See [Input Preconditions](prerequisite.md#input-preconditions) for the exact layout.

If your recording is a ROS1 bag containing `dvs_msgs/EventArray` messages or an iniVation AEDAT4 file, use the provided [data converters](data-conversion.md).

## 2. Choose a Runtime

### Native

Install the [required dependencies](prerequisite.md), then build from the repository root:

```bash
cmake -S . -B build -DCORNER2TAG_STANDALONE=ON
cmake --build build --parallel
```

Update `config/checkerboard.yaml` for your sensor, checkerboard, input file, and temporal window, then run:

```bash
./build/checkerboard config/checkerboard.yaml
```

### Docker

Install Docker, update the calibration fields in `docker/checkerboard.docker.yaml`, and run:

```bash
bash docker/run_checkerboard_docker.sh /absolute/path/to/data recording.h5
```

**Important:** Keep the space between `/absolute/path/to/data` and `recording.h5`. They are two separate arguments: the data directory and the HDF5 filename.

Keep `h5_path` and `out_dir` in the Docker configuration unchanged. The launcher maps the selected input and output locations automatically. See the [Docker guide](docker.md) for details.

## 3. Inspect the Results

Each run creates `results/checkerboard/run_YYYYMMDD_HHMMSS/`. Check the detected-corner images and calibration report before using `calibration.yaml`. The [Checkerboard Calibration](checkerboard.md#outputs) guide explains each output.
