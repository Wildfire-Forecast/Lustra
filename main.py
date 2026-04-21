import argparse


if __name__ == "__main__":
    print("[boot] Entering main.py...", flush=True)
    print("[boot] Importing LustraApp...", flush=True)
    from lustra.app import LustraApp

    parser = argparse.ArgumentParser(description="Run Lustra app")
    parser.add_argument("-v", "--verbose", action="store_true", help="Enable verbose mode and show all windows")
    args = parser.parse_args()

    print("[boot] Starting app...", flush=True)
    LustraApp(verbose=args.verbose).run()
