"""
Thin shim kept for backwards-compat / direct `python inference.py` usage.
The real implementation lives in app/inference.py.
"""
import argparse

from app.inference import run_inference  # noqa: F401  (re-exported)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run vocal removal on an audio file.")
    parser.add_argument("input_file", help="Path to the input audio file")
    parser.add_argument("output_dir", help="Directory where the output MP3 will be saved")
    parser.add_argument(
        "--device",
        default="cpu",
        help="Compute device: cpu | cuda | mps (default: cpu)",
    )
    args = parser.parse_args()

    run_inference(args.input_file, args.output_dir, device=args.device)
