"""
Stream OpenFace realtime CSV logs through Lab Streaming Layer (LSL).

Run `openface_realtime.py` to generate log rows, then start this script to
publish every row as a sample on an LSL outlet without modifying the
original realtime pipeline.
"""

import argparse
import csv
import os
import time
from typing import List, Optional, Sequence

from pylsl import StreamInfo, StreamOutlet, local_clock


def parse_args() -> argparse.Namespace:
    base_dir = os.path.dirname(__file__)
    default_logs = os.path.join(base_dir, "logs")

    parser = argparse.ArgumentParser(
        description=(
            "Tail the CSV logs emitted by openface_realtime.py and stream "
            "emotion, gaze, and AU values to Lab Streaming Layer."
        )
    )
    parser.add_argument(
        "--csv",
        dest="csv_path",
        help="Specific CSV file to stream. Defaults to the newest CSV inside the logs directory.",
    )
    parser.add_argument(
        "--log-dir",
        default=default_logs,
        help=f"Directory where session CSV files are stored (default: {default_logs}).",
    )
    parser.add_argument(
        "--stream-name",
        default="OpenFaceRealtime",
        help="Name of the published LSL stream.",
    )
    parser.add_argument(
        "--source-id",
        default="openface_realtime_csv",
        help="Unique source ID for the LSL stream.",
    )
    parser.add_argument(
        "--poll-interval",
        type=float,
        default=0.25,
        help="Seconds to wait between checks for new log rows.",
    )
    return parser.parse_args()


def wait_for_csv_file(log_dir: str, csv_path: Optional[str], poll_interval: float) -> str:
    """
    Wait until a CSV file is available. If csv_path is None, use the most recent
    CSV file in the log directory (matching *.csv).
    """
    if csv_path:
        target = os.path.abspath(csv_path)
        print(f"[bridge] Waiting for CSV file: {target}")
        while not os.path.exists(target):
            time.sleep(poll_interval)
        return target

    log_dir = os.path.abspath(log_dir)
    print(f"[bridge] Watching log directory: {log_dir}")
    while True:
        if os.path.isdir(log_dir):
            candidates = [
                os.path.join(log_dir, f)
                for f in os.listdir(log_dir)
                if f.lower().endswith(".csv")
            ]
            if candidates:
                latest = max(candidates, key=os.path.getmtime)
                print(f"[bridge] Using CSV file: {latest}")
                return latest
        time.sleep(poll_interval)


def wait_for_header_line(fh, poll_interval: float) -> Sequence[str]:
    """
    Block until a header row becomes available on the file handle.
    The function returns the parsed header and leaves the file pointer
    immediately after that line.
    """
    while True:
        line = fh.readline()
        if line:
            header = next(csv.reader([line]))
            print(f"[bridge] Header detected with {len(header)} columns.")
            return header
        time.sleep(poll_interval)


def pick_feature_columns(header: Sequence[str]) -> List[str]:
    ignore = {"frame_idx", "timestamp"}
    features = [col for col in header if col not in ignore]
    if not features:
        raise RuntimeError("CSV header does not contain numeric feature columns.")
    return features


def create_lsl_outlet(stream_name: str, source_id: str, feature_cols: Sequence[str]) -> StreamOutlet:
    info = StreamInfo(
        name=stream_name,
        type="Face",
        channel_count=len(feature_cols),
        nominal_srate=30,
        channel_format="float32",
        source_id=source_id,
    )
    chns = info.desc().append_child("channels")
    for col in feature_cols:
        ch = chns.append_child("channel")
        ch.append_child_value("label", col)
        ch.append_child_value("type", "feature")

    outlet = StreamOutlet(info)
    print(f"[bridge] LSL outlet created ({len(feature_cols)} channels).")
    return outlet


def convert_row(row: Sequence[str], header: Sequence[str], feature_cols: Sequence[str]) -> Optional[List[float]]:
    if len(row) != len(header):
        print(f"[bridge] Skipping malformed row with {len(row)} columns (expected {len(header)}).")
        return None

    mapping = dict(zip(header, row))
    sample: List[float] = []
    for col in feature_cols:
        try:
            sample.append(float(mapping[col]))
        except (KeyError, ValueError):
            print(f"[bridge] Unable to parse '{col}' -> '{mapping.get(col)}'; row skipped.")
            return None
    return sample


def tail_csv_and_stream(
    file_path: str,
    poll_interval: float,
    stream_name: str,
    source_id: str,
):
    """
    Follow the CSV file, convert every new row to floats, and push it onto LSL.
    """
    print(f"[bridge] Streaming from {file_path}. Press Ctrl+C to stop.")
    with open(file_path, "r", newline="") as fh:
        header = wait_for_header_line(fh, poll_interval)
        feature_cols = pick_feature_columns(header)
        outlet = create_lsl_outlet(stream_name, source_id, feature_cols)

        while True:
            line = fh.readline()
            if not line:
                time.sleep(poll_interval)
                continue

            row = next(csv.reader([line]))
            sample = convert_row(row, header, feature_cols)
            if sample is None:
                continue

            outlet.push_sample(sample, timestamp=local_clock())


def main():
    args = parse_args()
    csv_file = wait_for_csv_file(args.log_dir, args.csv_path, args.poll_interval)
    try:
        tail_csv_and_stream(
            file_path=csv_file,
            poll_interval=args.poll_interval,
            stream_name=args.stream_name,
            source_id=args.source_id,
        )
    except KeyboardInterrupt:
        print("\n[bridge] Streaming interrupted by user.")


if __name__ == "__main__":
    main()

