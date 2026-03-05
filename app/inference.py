import os
import tempfile
import time
from pathlib import Path


def run_inference(input_file: str, output_dir: str, device: str = "cpu") -> str:
    """
    Run vocal removal on input_file and write no_vocals.mp3 to output_dir.
    Returns the path of the output file.
    """
    from app.model import load_model
    from app.service.vocal_remover.runner import separate

    start = time.perf_counter()

    input_path = Path(input_file)
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_file}")

    audio_bytes = input_path.read_bytes()

    tmp_dir = Path(tempfile.mkdtemp())
    tmp_input = tmp_dir / input_path.name
    tmp_input.write_bytes(audio_bytes)

    tmp_output_dir = tmp_dir / "outputs"
    tmp_output_dir.mkdir(parents=True, exist_ok=True)

    print("Loading model (downloading if first run)...")
    model, torch_device = load_model(device=device)

    separate(
        input=str(tmp_input),
        model=model,
        device=torch_device,
        output_dir=str(tmp_output_dir),
        only_no_vocals=True,
    )

    basename = os.path.splitext(input_path.name)[0]
    no_vocals_path = tmp_output_dir / "vocal_remover" / basename / "no_vocals.mp3"

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    dest = out_dir / f"{basename}_no_vocals.mp3"
    dest.write_bytes(no_vocals_path.read_bytes())

    elapsed = time.perf_counter() - start
    print(f"Done in {elapsed:.2f}s — output: {dest}")
    return str(dest)
