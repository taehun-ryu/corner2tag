# Docker Quick Start

This guide shows the easiest way to run calibration with Docker.

## 1) Requirements
- Docker must be installed and available on your system.
- You need an input H5 file.
- Run the commands from the repository root directory.

## 2) Run
```bash
bash docker/run_checkerboard_docker.sh /absolute/path/to/data your_file.h5
```

Example:
```bash
bash docker/run_checkerboard_docker.sh /home/user/datasets/events sample.h5
```

## 3) Input Rules
- First argument: absolute path to the directory containing the H5 file.
- Second argument: H5 file name (or relative path inside that directory).

Valid example:
```bash
bash docker/run_checkerboard_docker.sh /data/session01 events.h5
```

:warning: Invalid example (second argument is absolute, so it will fail):
```bash
bash docker/run_checkerboard_docker.sh /data/session01 /data/session01/events.h5
```

## 4) Output Location
Results are created under the repository:

`results/checkerboard/run_YYYYMMDD_HHMMSS/`

## 5) Important Config Note
- :warning: When using Docker, please **do not edit** `h5_path` in `docker/checkerboard.docker.yaml`.
- Always choose the input file using the command-line argument (`h5_file_name`).
- Set `fixed_window_us` in `docker/checkerboard.docker.yaml` to a positive
  microsecond value before running.

## 6) GUI (Only If Needed)
GUI is disabled by default.

```bash
CORNER2TAG_ENABLE_GUI=1 \
bash docker/run_checkerboard_docker.sh /absolute/path/to/data your_file.h5
```

## 7) Common Errors
- `data directory not found`
  - Check that the first argument is a real directory.
- `expected input file not found`
  - Check that the file in the second argument exists under the first directory.
- `h5_file_name must be a path relative to data directory`
  - Use a file name or relative path, not an absolute path.

## Advanced Option (Optional)
- `CORNER2TAG_CHECKERBOARD_IMAGE_TAG`
  - Most users do not need this, because the default tag is applied automatically.
