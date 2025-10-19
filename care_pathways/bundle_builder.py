import json, hashlib
from datetime import datetime
from pathlib import Path

BASE = Path(__file__).resolve().parent
SRC = BASE / "source"
DIST = BASE / "dist"
DIST.mkdir(exist_ok=True)

def build_bundle(condition: str):
    """Build a NICE pathway bundle (merges pathway + metadata)."""
    pathway_file = SRC / condition / f"{condition}_pathway_enhanced_v3.json"
    metadata_file = SRC / condition / f"{condition}_pathway_enhanced_v3.metadata.json"

    if not pathway_file.exists() or not metadata_file.exists():
        raise FileNotFoundError("Pathway or metadata file missing!")

    pathway = json.load(open(pathway_file, encoding="utf-8"))
    metadata = json.load(open(metadata_file, encoding="utf-8"))

    bundle = {
        "bundle_id": f"{condition}_pathway_bundle",
        "version": "3.0.0",
        "build": {
            "built_at": datetime.utcnow().isoformat() + "Z",
            "built_by": "Uzair Madni",
            "source_files": [pathway_file.name, metadata_file.name],
        },
        "pathway": pathway,
        "metadata": metadata
    }

    # Calculate SHA256 hash
    content = json.dumps(bundle, sort_keys=True).encode("utf-8")
    bundle["build"]["content_sha256"] = hashlib.sha256(content).hexdigest()

    # Save bundle to dist/
    output_path = DIST / f"{condition}_pathway_bundle.v3.0.0.json"
    json.dump(bundle, open(output_path, "w", encoding="utf-8"), indent=2, ensure_ascii=False)

    print(f"✅ Bundle created: {output_path}")
    return output_path

if __name__ == "__main__":
    build_bundle("asthma")
