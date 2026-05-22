#!/usr/bin/env python3
"""
Render free-tier keep-warm pinger.

Render free-tier web services sleep after 15 minutes of inactivity, with a
30-60 second cold-start delay on the next request. For the live demo on
25 June 2026, run this from any laptop / Replit / a cron-like service so the
first request from the audience doesn't time out.

Usage:
    python scripts/keep_warm.py
    python scripts/keep_warm.py --url https://cardiosurv-api.onrender.com
    python scripts/keep_warm.py --interval 600   # 10 minutes (default)

Recommended free options to host this:
    * cron-job.org             (web-based, free, no signup needed for short runs)
    * UptimeRobot              (free monitoring, also keeps the service warm)
    * GitHub Actions schedule  (cron-style YAML, runs every 10 min)
"""
from __future__ import annotations

import argparse
import sys
import time
from datetime import datetime
from urllib.request import urlopen
from urllib.error import URLError, HTTPError


DEFAULT_URL = "https://cardiosurv-api.onrender.com/api/v1/health"
DEFAULT_INTERVAL = 600  # 10 minutes — well under Render's 15-min sleep threshold


def ping(url: str, timeout: int = 30) -> tuple[int, float]:
    t0 = time.time()
    try:
        with urlopen(url, timeout=timeout) as resp:
            elapsed = time.time() - t0
            return resp.status, elapsed
    except HTTPError as e:
        return e.code, time.time() - t0
    except URLError as e:
        print(f"  ! Network error: {e.reason}")
        return -1, time.time() - t0


def main() -> None:
    p = argparse.ArgumentParser(description="Keep Render free-tier API warm.")
    p.add_argument("--url", default=DEFAULT_URL,
                   help=f"Health endpoint URL (default: {DEFAULT_URL})")
    p.add_argument("--interval", type=int, default=DEFAULT_INTERVAL,
                   help=f"Seconds between pings (default: {DEFAULT_INTERVAL})")
    p.add_argument("--once", action="store_true",
                   help="Ping once and exit (useful for cron / GitHub Actions).")
    args = p.parse_args()

    print(f"[keep-warm] Target:   {args.url}")
    print(f"[keep-warm] Interval: {args.interval}s")
    print(f"[keep-warm] Mode:     {'one-shot' if args.once else 'loop'}")

    try:
        while True:
            stamp = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC")
            status, elapsed = ping(args.url)
            ok = "✓" if status == 200 else "✗"
            print(f"  [{stamp}] {ok} status={status} latency={elapsed*1000:.0f}ms")
            if args.once:
                sys.exit(0 if status == 200 else 1)
            time.sleep(args.interval)
    except KeyboardInterrupt:
        print("\n[keep-warm] Stopped.")


if __name__ == "__main__":
    main()
