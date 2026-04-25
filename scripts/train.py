#!/usr/bin/env python
"""Train a forecast model. Thin wrapper around ``meteo_hgt.runners.train``."""

from __future__ import annotations

import argparse

from meteo_hgt.runners import train


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True, help="Path to a YAML config (e.g. configs/variant3_hgt.yaml).")
    args = p.parse_args()
    train(args.config)


if __name__ == "__main__":
    main()
