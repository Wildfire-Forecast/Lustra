if __name__ == "__main__":
    print("[boot] Entering main.py...", flush=True)
    print("[boot] Importing LustraApp...", flush=True)
    from lustra.app import LustraApp

    print("[boot] Starting app...", flush=True)
    LustraApp().run()
