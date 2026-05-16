import argparse


if __name__ == "__main__":
    print("[boot] Entering main.py...", flush=True)
    print("[boot] Importing LustraApp...", flush=True)
    from lustra.app import LustraApp

    parser = argparse.ArgumentParser(description="Run Lustra app")
    parser.add_argument("-v", "--verbose", action="store_true", help="Enable verbose mode and show all windows")
    parser.add_argument(
        "--drone-height-m",
        type=float,
        default=18.0,
        help="Initial drone height in meters",
    )
    parser.add_argument(
        "--origin-lat",
        type=float,
        default=38.3700,
        help="Real-world latitude that the simulator origin maps to (default: DEÜ Tınaztepe, Buca/İzmir).",
    )
    parser.add_argument(
        "--origin-lon",
        type=float,
        default=27.2050,
        help="Real-world longitude that the simulator origin maps to (default: DEÜ Tınaztepe, Buca/İzmir).",
    )
    args = parser.parse_args()

    print("[boot] Starting app...", flush=True)
    LustraApp(
        verbose=args.verbose,
        drone_height_m=args.drone_height_m,
        origin_lat=args.origin_lat,
        origin_lon=args.origin_lon,
    ).run()
