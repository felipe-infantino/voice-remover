# voice-remover

Local vocal removal and audio source separation.

## Install

```bash
pip install voice-remover
```

## Usage

```bash
voice-remover "track.mp3" ./outputs/
voice-remover "track.mp3" ./outputs/ --device cuda
voice-remover "track.mp3" ./outputs/ --device mps
```

Output files are saved as `<track>_no_vocals.mp3` in the specified directory.

## Arguments

| Argument | Description |
|---|---|
| `input_file` | Path to the input audio file |
| `output_dir` | Directory where the output MP3 will be saved |

## Options

| Flag | Default | Description |
|---|---|---|
| `--device` | `cpu` | Inference device: `cpu`, `cuda`, `mps` |

## Requirements

- Python 3.11–3.14
- FFmpeg (`brew install ffmpeg` / `apt install ffmpeg`)

> Model weights download automatically on first run from Hugging Face (`felipeinfantino/voice-remover`).
> Cached at `~/.cache/voice-remover/models/`.

## Development

```bash
git clone https://github.com/felipeinfantino/voice-remover
cd voice-remover
poetry install
poetry run voice-remover "track.mp3" ./outputs/
```

### Extending with new pagackage 
 
```bash
poetry add [packagename]

#check the cli is still working
poetry run voice-remover "track.mp3" ./outputs/

# Verify lockfile is clean
poetry lock

# Bump version
poetry version patch   # or minor / major

```

## License

MIT
