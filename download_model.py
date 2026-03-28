"""
Download the fine-tuned barbell YOLO model from Roboflow and save it
as models/barbell.pt for use by core/barbell.py.

Usage
-----
    pip install roboflow
    python download_model.py --api-key YOUR_ROBOFLOW_API_KEY

Optional flags
--------------
    --version INT   Model version to download (default: 1)
    --format  STR   Export format: yolov8 | yolov5pytorch (default: yolov8)

After running, the app will automatically pick up models/barbell.pt.
"""
import argparse
import glob
import shutil
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="Download barbell YOLO model from Roboflow")
    parser.add_argument("--api-key", required=True, help="Roboflow API key")
    parser.add_argument("--version", type=int, default=1, help="Model version (default: 1)")
    parser.add_argument(
        "--format", default="yolov11",
        choices=["yolov11", "yolov8", "yolov5pytorch"],
        help="Export format (default: yolov11)",
    )
    args = parser.parse_args()

    try:
        from roboflow import Roboflow
    except ImportError:
        print("roboflow package not found. Install it first:\n  pip install roboflow")
        return

    print(f"Connecting to Roboflow (project: barbell-zwl3l-ambrq, version: {args.version}) ...")
    rf      = Roboflow(api_key=args.api_key)
    project = rf.workspace("workspace-ftm9u").project("barbell-zwl3l-ambrq")
    version = project.version(args.version)
    version.download(args.format)

    # Locate the downloaded .pt file
    pt_files = glob.glob("barbell-zwl3l-ambrq-*/**/*.pt", recursive=True)
    if not pt_files:
        # Fallback: search current directory tree
        pt_files = glob.glob("**/*.pt", recursive=True)

    if not pt_files:
        print(
            "\nCould not locate the downloaded .pt file automatically.\n"
            "Please copy it manually to:  models/barbell.pt"
        )
        return

    models_dir = Path("models")
    models_dir.mkdir(exist_ok=True)
    dest = models_dir / "barbell.pt"
    shutil.copy(pt_files[0], dest)
    print(f"\nModel saved to {dest}")
    print("You can now run:  python app.py")


if __name__ == "__main__":
    main()
