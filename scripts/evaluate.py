#!/usr/bin/env python
"""Evaluate a trained checkpoint on a held-out partition."""

from __future__ import annotations

import argparse

from meteo_hgt.runners import evaluate


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--partition", default="test", choices=["train", "val", "test"])
    args = p.parse_args()
    evaluate(args.config, args.checkpoint, partition=args.partition)


if __name__ == "__main__":
    main()
