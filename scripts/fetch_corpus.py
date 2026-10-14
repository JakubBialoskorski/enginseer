"""Download the general pretraining corpus listed under [[corpus]] in sources.toml into data/corpus/<name>/.

data/corpus/manifest.json records the dataset revision and files of every corpus for the model card.
"""

import argparse
import json
import tomllib
from datetime import date

import paths  # noqa: F401  (sets the HF cache location)
import pyarrow.parquet as pq
from huggingface_hub import HfApi, snapshot_download
from paths import DATA, ROOT

CORPUS = DATA / "corpus"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--only", nargs="*", help="fetch only these corpus names")
    args = parser.parse_args()

    corpora = tomllib.loads((ROOT / "sources.toml").read_text())["corpus"]
    manifest_path = CORPUS / "manifest.json"
    CORPUS.mkdir(parents=True, exist_ok=True)
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}

    for corpus in corpora:
        if args.only and corpus["name"] not in args.only:
            continue
        print(corpus["name"])
        revision = HfApi().dataset_info(corpus["dataset"]).sha
        dest = CORPUS / corpus["name"]
        snapshot_download(corpus["dataset"], repo_type="dataset", revision=revision,
                          allow_patterns=corpus["files"], local_dir=dest)
        files = sorted(p for p in dest.rglob("*.parquet"))
        rows = sum(pq.ParquetFile(p).metadata.num_rows for p in files)
        manifest[corpus["name"]] = {
            "dataset": corpus["dataset"], "revision": revision, "license": corpus["license"],
            "files": [p.relative_to(dest).as_posix() for p in files], "rows": rows,
            "fetched": date.today().isoformat(),
        }
        manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
        print(f"  {len(files)} files, {rows:,} rows")


if __name__ == "__main__":
    main()
