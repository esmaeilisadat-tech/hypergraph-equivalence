"""Hash the published Git tree, excluding the manifest's own recursive hash."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "docs/github/publication_manifest.json"


def digest(path):
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024*1024), b""):
            value.update(chunk)
    return value.hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--write", action="store_true")
    args = parser.parse_args()
    if args.write:
        names = subprocess.check_output(["git", "ls-files", "-z"],cwd=ROOT).decode().split("\0")
        entries = [{"path":name,"size_bytes":(ROOT/name).stat().st_size,"sha256":digest(ROOT/name)}
                   for name in sorted(set(names)) if name and ROOT/name != MANIFEST]
        MANIFEST.write_text(json.dumps({"scope":"Published repository files; excludes this manifest to avoid a recursive hash", "entries":entries},indent=2)+"\n",encoding="utf-8")
    data = json.loads(MANIFEST.read_text(encoding="utf-8"))
    failures = [entry["path"] for entry in data["entries"] if not (ROOT/entry["path"]).is_file()
                or (ROOT/entry["path"]).stat().st_size != entry["size_bytes"]
                or digest(ROOT/entry["path"]) != entry["sha256"]]
    print(json.dumps({"status":"PASS" if not failures else "FAIL","checked_files":len(data["entries"]),"failures":failures},indent=2))
    raise SystemExit(bool(failures))


if __name__ == "__main__":
    main()
