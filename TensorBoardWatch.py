"""
Tails metrics.jsonl written by autoresearch-cpp and pushes scalars
to TensorBoard in real time.

Usage:
    # terminal 1 — start training
    ./build/autoresearch --device cpu --data-dir data --budget-secs 300

    # terminal 2 — start watcher
    python tensorboard_watch.py

    # terminal 3 — start TensorBoard
    tensorboard --logdir runs/

Dependencies:
    pip install tensorboard
"""

import argparse
import json
import sys
import time
from pathlib import Path

try:
    from torch.utils.tensorboard import SummaryWriter
except ImportError:
    sys.exit("tensorboard not found. Run: pip install tensorboard")


def tail(path: Path, poll_interval: float = 0.5):
    """Yields new lines from path as they are appended, blocking between polls."""
    with open(path, "r") as f:
        # Seek to end so we only see new lines written after this process starts.
        f.seek(0, 2)
        while True:
            line = f.readline()
            if line:
                yield line
            else:
                time.sleep(poll_interval)


def main() -> None:
    parser = argparse.ArgumentParser(description="Push autoresearch-cpp metrics to TensorBoard.")
    parser.add_argument("--metrics-file", default="metrics.jsonl",
                        help="Path to the metrics JSONL file (default: metrics.jsonl)")
    parser.add_argument("--log-dir",      default="runs",
                        help="TensorBoard log directory (default: runs)")
    parser.add_argument("--poll",         type=float, default=0.25,
                        help="File poll interval in seconds (default: 0.25)")
    args = parser.parse_args()

    metrics_path = Path(args.metrics_file)

    # Wait for the file to appear if training hasn't started yet.
    print(f"Waiting for {metrics_path}...")
    while not metrics_path.exists():
        time.sleep(0.5)

    writer = SummaryWriter(log_dir=args.log_dir)
    print(f"Writing to TensorBoard run dir: {args.log_dir}")
    print(f"Start TensorBoard with:  tensorboard --logdir {args.log_dir}")
    print("Watching for metrics... (Ctrl-C to stop)")

    try:
        for raw_line in tail(metrics_path, poll_interval=args.poll):
            line = raw_line.strip()
            if not line:
                continue

            try:
                entry = json.loads(line)
            except json.JSONDecodeError:
                continue

            step       = entry.get("step", 0)
            train_loss = entry.get("train_loss", 0.0)
            val_bpb    = entry.get("val_bpb",    0.0)
            elapsed_s  = entry.get("elapsed_s",  0.0)

            if train_loss > 0.0:
                writer.add_scalar("train/loss",    train_loss, global_step=step, walltime=elapsed_s)

            if val_bpb > 0.0:
                writer.add_scalar("val/bpb",       val_bpb,    global_step=step, walltime=elapsed_s)

            writer.flush()

    except KeyboardInterrupt:
        print("\nStopped.")
    finally:
        writer.close()


if __name__ == "__main__":
    main()