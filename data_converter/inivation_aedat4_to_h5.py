#!/usr/bin/env python3
import os
import glob
import argparse
import h5py
import numpy as np
import dv_processing as dv

def save_events_h5(timestamps, xs, ys, ps, h5_path):
    os.makedirs(os.path.dirname(h5_path) or ".", exist_ok=True)

    with h5py.File(h5_path, "w") as hf:
        grp = hf.create_group("events")
        grp.create_dataset("ts",
                           data=np.array(timestamps, dtype=np.uint64),
                           compression="gzip", chunks=True)
        grp.create_dataset("xs",
                           data=np.array(xs, dtype=np.uint16),
                           compression="gzip", chunks=True)
        grp.create_dataset("ys",
                           data=np.array(ys, dtype=np.uint16),
                           compression="gzip", chunks=True)
        grp.create_dataset("ps",
                           data=np.array(ps, dtype=np.uint8),
                           compression="gzip", chunks=True)

    print(f"[+] Saved HDF5 events to: {h5_path}")


def resolve_output_path(input_path, output_dir=None, output_name=None):
    base = output_name or os.path.splitext(os.path.basename(input_path))[0]
    if base.endswith(".h5"):
        base = os.path.splitext(base)[0]
    out_dir = output_dir or os.path.dirname(os.path.abspath(input_path))
    return os.path.join(out_dir, f"{base}.h5")


def convert_aedat4_to_h5(aedat4_path, h5_path, zero_ts=False):
    reader = dv.io.MonoCameraRecording(aedat4_path)
    print(f"[+] Opened {aedat4_path} (camera: {reader.getCameraName()})")

    timestamps, xs, ys, ps = [], [], [], []
    while reader.isRunning():
        batch = reader.getNextEventBatch()
        if batch is None:
            continue
        for e in batch:
            t_microseconds = e.timestamp()
            timestamps.append(t_microseconds)
            xs.append(e.x())
            ys.append(e.y())
            ps.append(1 if e.polarity() else 0)

    if len(timestamps) == 0:
        print(f"[-] No events read from {aedat4_path}; skipping.")
        return

    if zero_ts:
        t0 = timestamps[0]
        timestamps = [t - t0 for t in timestamps]
        print(f"[+] Zero-based timestamps applied (t0 = {t0})")

    save_events_h5(timestamps, xs, ys, ps, h5_path=h5_path)


def main():
    parser = argparse.ArgumentParser(
        description="Convert AEDAT4 events to HDF5 (events/{ts,xs,ys,ps})"
    )
    parser.add_argument("input", help="Input .aedat4 file or directory")
    parser.add_argument(
        "--output_dir", "-o", default=None,
        help="Output directory for .h5 (default: same directory as each input file)"
    )
    parser.add_argument(
        "--output_name", "-n", default=None,
        help="Output filename without extension (single input only)"
    )
    parser.add_argument(
        "--zero_ts", "-z", action="store_true",
        help="Subtract first timestamp so that ts[0] == 0"
    )
    args = parser.parse_args()

    if os.path.isdir(args.input):
        aedat4_files = sorted(glob.glob(os.path.join(args.input, "*.aedat4")))
    else:
        aedat4_files = [args.input]

    if len(aedat4_files) == 0:
        print(f"[-] No .aedat4 files found in: {args.input}")
        return

    is_single_input = len(aedat4_files) == 1
    if args.output_name and not is_single_input:
        print("[!] --output_name is only applied for a single input file; ignoring it.")

    for aedat4_file in aedat4_files:
        output_name = args.output_name if is_single_input else None
        h5_path = resolve_output_path(
            aedat4_file,
            output_dir=args.output_dir,
            output_name=output_name
        )
        print(f"Extracting events from {aedat4_file} -> {h5_path}")
        convert_aedat4_to_h5(aedat4_file, h5_path, zero_ts=args.zero_ts)

if __name__ == "__main__":
    main()
