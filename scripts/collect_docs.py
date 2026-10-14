"""Fetch the documentation listed in sources.toml into data/raw/ as shallow, sparse git clones.

data/raw/manifest.json records the fetched commit of every source for the model card.
"""

import argparse
import json
import subprocess
import tomllib
from datetime import date

from paths import DATA, ROOT

RAW = DATA / "raw"


def git(*args, cwd=None):
    return subprocess.run(["git", *args], cwd=cwd, check=True, capture_output=True, text=True).stdout.strip()


def fetch_git(src):
    dest = RAW / src["name"]
    if not dest.exists():
        git("clone", "--depth", "1", "--filter=blob:none", "--sparse", src["url"], str(dest))
    else:
        git("fetch", "--depth", "1", "origin", "HEAD", cwd=dest)
        git("reset", "--hard", "FETCH_HEAD", cwd=dest)
    if src["paths"]:
        git("sparse-checkout", "set", *src["paths"], cwd=dest)
    else:
        git("sparse-checkout", "disable", cwd=dest)
    return {"commit": git("rev-parse", "HEAD", cwd=dest)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--only", nargs="*", help="fetch only these source names")
    args = parser.parse_args()

    sources = tomllib.loads((ROOT / "sources.toml").read_text())["source"]
    manifest_path = RAW / "manifest.json"
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
    RAW.mkdir(parents=True, exist_ok=True)

    for src in sources:
        if args.only and src["name"] not in args.only:
            continue
        print(src["name"])
        fetched = fetch_git(src)
        manifest[src["name"]] = {
            key: src[key] for key in ("domain", "license", "url", "paths", "note") if key in src
        } | fetched | {"fetched": date.today().isoformat()}
        manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")


if __name__ == "__main__":
    main()
