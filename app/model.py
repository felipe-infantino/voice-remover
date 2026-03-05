from pathlib import Path

_model = None
_device = None


def get_model_path() -> Path:
    from huggingface_hub import hf_hub_download
    from app.config import CACHE_DIR, MODEL_FILENAME, MODEL_ID, MODEL_REVISION

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    return Path(
        hf_hub_download(
            repo_id=MODEL_ID,
            filename=MODEL_FILENAME,
            revision=MODEL_REVISION,
            cache_dir=str(CACHE_DIR),
        )
    )


def load_model(device: str = "cpu"):
    """Download (once) and load the model, caching the result for the process lifetime."""
    global _model, _device

    if _model is not None:
        return _model, _device

    import torch
    from app.service.vocal_remover import nets

    model_path = get_model_path()
    torch_device = torch.device(device)

    model = nets.CascadedNet(n_fft=2048, nout=32, nout_lstm=128)
    model.load_state_dict(
        torch.load(model_path, map_location=torch_device, weights_only=True)
    )
    model.to(torch_device)
    model.eval()

    _model = model
    _device = torch_device
    return _model, _device
