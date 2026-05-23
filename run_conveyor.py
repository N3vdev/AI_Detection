"""
Conveyor Belt Live Inspection System

Usage:
    python run_conveyor.py
    python run_conveyor.py --max 50          # stop after 50 products
    python run_conveyor.py --session my_id   # custom session name
"""
import argparse
import datetime
import sys
import uuid

from live.conveyor_main import ConveyorSystem


def main():
    parser = argparse.ArgumentParser(description="Conveyor Belt AI Inspection")
    parser.add_argument("--max", type=int, default=None,
                        help="Max products to inspect (default: from config)")
    parser.add_argument("--session", type=str, default=None,
                        help="Session ID (default: auto-generated)")
    args = parser.parse_args()

    session_id = args.session or (
        f"truck_{datetime.date.today().isoformat()}_{uuid.uuid4().hex[:6]}"
    )

    system = ConveyorSystem()
    system.start(session_id)
    try:
        system.run_session(max_products=args.max)
    finally:
        system.stop()


if __name__ == "__main__":
    main()
