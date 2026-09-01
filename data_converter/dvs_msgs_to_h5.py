#!/usr/bin/env python3
import os
import glob
import argparse

import rosbag
import h5py
import numpy as np
from tqdm import tqdm

def ts_us_int(t):
    return int(t.secs) * 1_000_000 + int(t.nsecs) // 1_000


def resolve_output_path(input_path, output_dir=None, output_name=None):
    base = output_name or os.path.splitext(os.path.basename(input_path))[0]
    if base.endswith(".h5"):
        base = os.path.splitext(base)[0]
    out_dir = output_dir or os.path.dirname(os.path.abspath(input_path))
    return os.path.join(out_dir, f"{base}.h5")

def extract_ros1_to_dv_h5(bag_path, h5_path, event_topic, zero_ts=False, buffer_events=2_000_000):
    if not os.path.exists(bag_path):
        raise FileNotFoundError(bag_path)

    os.makedirs(os.path.dirname(h5_path) or ".", exist_ok=True)

    with h5py.File(h5_path, "w") as hf:
        grp = hf.create_group("events")

        ds_t = grp.create_dataset(
            "ts",
            shape=(0,),
            maxshape=(None,),
            dtype=np.uint64,
            compression="gzip",
            chunks=True
        )
        ds_x = grp.create_dataset(
            "xs",
            shape=(0,),
            maxshape=(None,),
            dtype=np.uint16,
            compression="gzip",
            chunks=True
        )
        ds_y = grp.create_dataset(
            "ys",
            shape=(0,),
            maxshape=(None,),
            dtype=np.uint16,
            compression="gzip",
            chunks=True
        )
        ds_p = grp.create_dataset(
            "ps",
            shape=(0,),
            maxshape=(None,),
            dtype=np.uint8,
            compression="gzip",
            chunks=True
        )

        def append(ds, arr):
            n = int(arr.shape[0])
            if n == 0:
                return
            old = int(ds.shape[0])
            ds.resize((old + n,))
            ds[old:old + n] = arr

        first_ts_us = None

        buf_x = []
        buf_y = []
        buf_t = []
        buf_p = []
        buf_count = 0
        total_events = 0

        def flush():
            nonlocal buf_x, buf_y, buf_t, buf_p, buf_count
            if buf_count == 0:
                return

            x = np.concatenate(buf_x, axis=0)
            y = np.concatenate(buf_y, axis=0)
            t = np.concatenate(buf_t, axis=0)
            p = np.concatenate(buf_p, axis=0)

            append(ds_x, x)
            append(ds_y, y)
            append(ds_t, t)
            append(ds_p, p)

            buf_x = []
            buf_y = []
            buf_t = []
            buf_p = []
            buf_count = 0

        with rosbag.Bag(bag_path, "r") as bag:
            it = bag.read_messages(topics=[event_topic])

            for _, msg, _ in tqdm(it, desc="Reading event messages"):
                if not hasattr(msg, "events"):
                    continue
                if getattr(msg, "_type", "") not in ("dvs_msgs/EventArray", ""):
                    continue
                if len(msg.events) == 0:
                    continue

                n = len(msg.events)

                if first_ts_us is None:
                    first_ts_us = ts_us_int(msg.events[0].ts)

                xs = np.fromiter((e.x for e in msg.events), dtype=np.uint16, count=n)
                ys = np.fromiter((e.y for e in msg.events), dtype=np.uint16, count=n)

                secs = np.fromiter((e.ts.secs for e in msg.events), dtype=np.int64, count=n)
                nsecs = np.fromiter((e.ts.nsecs for e in msg.events), dtype=np.int64, count=n)
                ts_us = secs * 1_000_000 + nsecs // 1_000

                if zero_ts:
                    ts_us = ts_us - first_ts_us

                ts = ts_us.astype(np.uint64, copy=False)

                ps = np.fromiter((1 if e.polarity else 0 for e in msg.events), dtype=np.uint8, count=n)

                buf_x.append(xs)
                buf_y.append(ys)
                buf_t.append(ts)
                buf_p.append(ps)
                buf_count += n
                total_events += n

                if buf_count >= buffer_events:
                    flush()

        flush()

    print(f"✓ Wrote {total_events} events to {h5_path}")


def main():
    parser = argparse.ArgumentParser(description="ROS1 bag → HDF5 (DV-compatible format: events/{ts,xs,ys,ps})")
    parser.add_argument("input", help="rosbag file or directory")
    parser.add_argument(
        "--event_topic", "--topic", "-t", dest="event_topic",
        default="/dvs/events", help="event topic name"
    )
    parser.add_argument(
        "--output_dir", "-o", default=None,
        help="output directory for .h5 (default: same directory as each input bag)"
    )
    parser.add_argument(
        "--output_name", "-n", default=None,
        help="output filename without extension (single input only)"
    )
    parser.add_argument("--zero_ts", "-z", action="store_true")
    parser.add_argument("--buffer_events", type=int, default=2_000_000)
    args = parser.parse_args()

    if os.path.isdir(args.input):
        bags = sorted(glob.glob(os.path.join(args.input, "*.bag")))
    else:
        bags = [args.input]

    if len(bags) == 0:
        print(f"[-] No .bag files found in: {args.input}")
        return

    is_single_input = len(bags) == 1
    if args.output_name and not is_single_input:
        print("[!] --output_name is only applied for a single input bag; ignoring it.")

    for bag_path in bags:
        output_name = args.output_name if is_single_input else None
        h5_path = resolve_output_path(
            bag_path,
            output_dir=args.output_dir,
            output_name=output_name
        )

        print(f"Extracting events from {bag_path} → {h5_path}")
        extract_ros1_to_dv_h5(
            bag_path,
            h5_path,
            args.event_topic,
            zero_ts=args.zero_ts,
            buffer_events=args.buffer_events
        )


if __name__ == "__main__":
    main()
