from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
TRAIN_SCRIPT = PROJECT_ROOT / "scripts" / "train_multimodal.py"
RESULTS_DIR = PROJECT_ROOT / "artifacts" / "results"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run multimodal training sequentially one fold at a time.")
    parser.add_argument("--experiment-name", required=True)
    parser.add_argument("--folds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--subject-weighting", dest="subject_weighting", action="store_true")
    parser.add_argument("--no-subject-weighting", dest="subject_weighting", action="store_false")
    parser.set_defaults(subject_weighting=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result_root = RESULTS_DIR / args.experiment_name
    result_root.mkdir(parents=True, exist_ok=True)

    for fold_idx in args.folds:
        cmd = [
            sys.executable,
            str(TRAIN_SCRIPT),
            "--experiment-name",
            args.experiment_name,
            "--folds",
            str(fold_idx),
            "--device",
            args.device,
        ]
        if args.local_files_only:
            cmd.append("--local-files-only")
        if args.resume:
            cmd.append("--resume")
        if args.subject_weighting:
            cmd.append("--subject-weighting")
        else:
            cmd.append("--no-subject-weighting")

        log_path = result_root / f"fold_{fold_idx}_train.log"
        with log_path.open("a", encoding="utf-8") as log_file:
            log_file.write(f"\n=== fold {fold_idx} ===\n")
            log_file.write("COMMAND: " + " ".join(cmd) + "\n")
            log_file.flush()

            process = subprocess.Popen(
                cmd,
                cwd=str(PROJECT_ROOT),
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
            )

            assert process.stdout is not None
            for line in process.stdout:
                sys.stdout.write(line)
                sys.stdout.flush()
                log_file.write(line)
                log_file.flush()

            return_code = process.wait()
            if return_code != 0:
                raise subprocess.CalledProcessError(return_code, cmd)


if __name__ == "__main__":
    main()
