#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

from benchmark_contracts import DEFAULT_SCENARIO_OUTPUT, compile_contracts_to_dataset, load_contracts, write_dataset


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Compile benchmark contract JSON files into runtime conversation-eval scenarios."
    )
    parser.add_argument("input", help="Benchmark contract JSON file or directory.")
    parser.add_argument(
        "--output",
        default=DEFAULT_SCENARIO_OUTPUT,
        help="Output runtime scenario JSON file path.",
    )
    args = parser.parse_args()

    contracts = load_contracts(Path(args.input))
    dataset = compile_contracts_to_dataset(contracts)
    output_path = write_dataset(Path(args.output), dataset)
    print(output_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
