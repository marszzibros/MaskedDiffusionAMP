"""
Run the full DBAASP species download.

Creates the directories the pipeline writes to, then downloads the peptide IDs
and per-species details into CSV. Peptides that could NOT be saved are recorded,
one per line, in log/dbaasp/dbaasp.log (tab-separated: NOT SAVED <id>
<sequence> <reason>).
"""

import argparse
import os
from pathlib import Path

from dbaasp_download import DBAASP


def main():
    parser = argparse.ArgumentParser(description="Download the DBAASP species dataset.")
    parser.add_argument(
        "--id-path",
        default=None,
        help="Path to an existing DBAASP id list. If omitted, ids are downloaded.",
    )
    parser.add_argument(
        "--detail-path",
        default=None,
        help="Path to an existing details CSV. If omitted, details are downloaded.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=10,
        help="Batch size per worker for multiprocessing (default: 10).",
    )
    args = parser.parse_args()

    # run relative to the project root so all data/log paths resolve the same
    # way no matter where the script is launched from
    os.chdir(Path(__file__).resolve().parents[1])

    # The pipeline writes to these locations but does not create them itself.
    for directory in ("dataset/data/dbaasp", "log/dbaasp"):
        os.makedirs(directory, exist_ok=True)

    log_path = "log/dbaasp/dbaasp.log"
    # start each run with a fresh failure log so it reflects only this run
    open(log_path, "w").close()

    print("Starting DBAASP species download...")
    DBAASP(
        id_path=args.id_path,
        detail_path=args.detail_path,
        batch_size=args.batch_size,
    )
    print("Done.")

    # report where the not-saved peptides ended up
    if os.path.exists(log_path):
        with open(log_path) as fh:
            failures = [line for line in fh if line.startswith("NOT SAVED")]
        print(f"{len(failures)} peptide(s) were NOT saved. See {log_path}")
    else:
        print(f"No failure log was produced at {log_path}")


if __name__ == "__main__":
    main()
