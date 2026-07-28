"""CLI entry point for deterministic Phase 3 split generation."""

from src.phase3_data import write_all_splits


def main() -> None:
    hashes = write_all_splits()
    for split_name, digest in hashes.items():
        print(f"{split_name}: {digest}")


if __name__ == "__main__":
    main()
