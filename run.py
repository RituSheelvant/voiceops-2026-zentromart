from src.voiceops import main

if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (RuntimeError, FileNotFoundError) as exc:
        import sys
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(2)
