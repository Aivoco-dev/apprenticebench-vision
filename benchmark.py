"""CLI for the Vision Large accounts-payable runner."""

import argparse
import importlib
import json
import os
import sys
from decimal import Decimal, InvalidOperation
from pathlib import Path

from runner import RunError, run
from vision import Vision, VisionError


def api_key():
    value = os.environ.get("VISION_API_KEY")
    if value and value.strip():
        return value.strip()

    env_file = Path(".env")
    if env_file.is_file():
        for line in env_file.read_text(encoding="utf-8").splitlines():
            key, separator, value = line.partition("=")
            if separator and key.strip() == "VISION_API_KEY":
                value = value.strip().strip('"').strip("'")
                if value:
                    return value
    raise RunError("Set VISION_API_KEY in .env or export it in the environment")


def factory(path):
    module, separator, name = path.partition(":")
    if not separator or not module or not name:
        raise RunError("Specify the environment as module:factory")
    try:
        create = getattr(importlib.import_module(module), name)
    except (ImportError, AttributeError) as error:
        raise RunError("Could not load the requested environment factory") from error
    if not callable(create):
        raise RunError("The environment factory must be callable")
    return create


def positive_int(value):
    number = int(value)
    if number <= 0:
        raise argparse.ArgumentTypeError("must be positive")
    return number


def positive_cost(value):
    try:
        amount = Decimal(value)
    except InvalidOperation:
        raise argparse.ArgumentTypeError("must be a positive dollar amount") from None
    if not amount.is_finite() or amount <= 0:
        raise argparse.ArgumentTypeError("must be a positive dollar amount")
    return amount


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--environment", required=True, metavar="MODULE:FACTORY",
                        help="trusted Python adapter for the evaluation environment")
    parser.add_argument("--report", type=Path, required=True,
                        help="new path for the JSON report")
    parser.add_argument("--max-calls-per-task", type=positive_int, default=150)
    parser.add_argument("--max-estimated-usd", type=positive_cost,
                        help="stop before the next request once the estimate reaches this amount")
    args = parser.parse_args(argv)

    try:
        if args.report.exists():
            raise RunError("Report already exists. Use a new report path.")
        create = factory(args.environment)
        vision = Vision(api_key())
        report = run(create(), vision, args.report, max_calls=args.max_calls_per_task,
                     max_usd=args.max_estimated_usd)
    except (RunError, VisionError, OSError) as error:
        print(f"Run stopped ({type(error).__name__}): {error}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        print("Run interrupted", file=sys.stderr)
        return 130

    summary = {key: value for key, value in report.items() if key != "tasks"}
    print(json.dumps(summary, indent=2))
    print(f"Report saved to {args.report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
