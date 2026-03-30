
import argparse
import subprocess
import sys

def seconds_to_hms(s: int) -> str:
    h  = s // 3600
    m  = (s % 3600) // 60
    sec = s % 60
    return f"{h:02d}:{m:02d}:{sec:02d}"

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--url",      required=True)
    parser.add_argument("--start",    type=int, default=0)
    parser.add_argument("--duration", type=int, default=50)
    parser.add_argument("--out",      default="test_video.mp4")
    args = parser.parse_args()

    t_start = seconds_to_hms(args.start)
    t_end   = seconds_to_hms(args.start + args.duration)

    print(f"Downloading  {args.url}")
    print(f"Trimming     {t_start} → {t_end}  →  {args.out}\n")

    cmd = [
        "yt-dlp",
        "--format", "bestvideo[ext=mp4][height<=720]+bestaudio[ext=m4a]/best[ext=mp4]/best",
        "--merge-output-format", "mp4",
        "--download-sections", f"*{t_start}-{t_end}",
        "--force-keyframes-at-cuts",
        "--no-playlist",
        "--output", args.out,
        args.url,
    ]

    result = subprocess.run(cmd)
    if result.returncode != 0:
        print("\nFailed. Try without --force-keyframes-at-cuts if ffmpeg is missing.")
        sys.exit(1)

    print(f"\nDone! Saved to: {args.out}")
    print("Upload this file to the 'Upload Video' tab in the app.")

if __name__ == "__main__":
    main()
