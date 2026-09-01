# Data Conversion

corner2tag reads event data from HDF5. Two converters are included for recordings that use other formats:

| Input | Converter |
| --- | --- |
| ROS1 bag with `dvs_msgs/EventArray` | `data_converter/dvs_msgs_to_h5.py` |
| iniVation AEDAT4 | `data_converter/inivation_aedat4_to_h5.py` |

Both converters write the [HDF5 layout required by corner2tag](prerequisite.md#input-preconditions). Timestamps are stored as integer microseconds, coordinates as unsigned 16-bit integers, and polarity as `0` or `1`.

!!! warning
    If the destination `.h5` file already exists, the converter replaces it. Choose the output path carefully.

## ROS1 Bag

### Requirements

Run the converter in a ROS1 environment that can deserialize the bag's `dvs_msgs/EventArray` messages. The Python environment must provide:

- `rosbag` (including the message definitions used by the recording)
- `h5py`
- `numpy`
- `tqdm`

Convert one bag using the default event topic, `/dvs/events`:

```bash
python3 data_converter/dvs_msgs_to_h5.py /path/to/recording.bag
```

Use `--event_topic` when the events were recorded under another topic:

```bash
python3 data_converter/dvs_msgs_to_h5.py /path/to/recording.bag \
  --event_topic /camera/events
```

By default, the output is written next to the input with the same basename and an `.h5` extension. Use `--output_dir` or, for a single input, `--output_name` to change it:

```bash
python3 data_converter/dvs_msgs_to_h5.py /path/to/recording.bag \
  --output_dir /path/to/converted \
  --output_name calibration_events
```

Passing a directory converts every `.bag` file directly inside that directory:

```bash
python3 data_converter/dvs_msgs_to_h5.py /path/to/bag_directory \
  --output_dir /path/to/converted
```

The converter buffers events before appending them to HDF5. The default is 2,000,000 events; reduce it if memory is constrained:

```bash
python3 data_converter/dvs_msgs_to_h5.py /path/to/recording.bag \
  --buffer_events 500000
```

## iniVation AEDAT4

### Requirements

The Python environment must provide:

- `dv-processing` with its Python bindings
- `h5py`
- `numpy`

Convert one recording:

```bash
python3 data_converter/inivation_aedat4_to_h5.py /path/to/recording.aedat4
```

Output naming and directory conversion work in the same way as the ROS1 converter:

```bash
python3 data_converter/inivation_aedat4_to_h5.py /path/to/aedat4_directory \
  --output_dir /path/to/converted
```

!!! note
    The AEDAT4 converter currently accumulates the recording's events in memory before writing HDF5. Make sure sufficient memory is available for long recordings.

## Timestamp Origin

Both converters preserve the recording timestamps by default. Add `--zero_ts` to subtract the first event timestamp:

```bash
python3 data_converter/inivation_aedat4_to_h5.py /path/to/recording.aedat4 \
  --zero_ts
```

corner2tag internally anchors fixed windows to the first event, so zero-based input timestamps are optional.

## Check the Available Options

Each converter exposes its complete command-line interface through `--help`:

```bash
python3 data_converter/dvs_msgs_to_h5.py --help
python3 data_converter/inivation_aedat4_to_h5.py --help
```

After conversion, set the generated file as `h5_path` for a native run or pass its directory and filename to the [Docker launcher](docker.md#run).
