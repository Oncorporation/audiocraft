# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
Utility functions to load from the checkpoints.
Each checkpoint is a torch.saved dict with the following keys:
- 'xp.cfg': the hydra config as dumped during training. This should be used
    to rebuild the object using the audiocraft.models.builders functions,
- 'model_best_state': a readily loadable best state for the model, including
    the conditioner. The model obtained from `xp.cfg` should be compatible
    with this state dict. In the case of a LM, the encodec model would not be
    bundled along but instead provided separately.

Those functions also support loading from a remote location with the Torch Hub API.
They also support overriding some parameters, in particular the device and dtype
of the returned model.
"""

from pathlib import Path
from huggingface_hub import hf_hub_download
import typing as tp
import os

from omegaconf import OmegaConf, DictConfig
import torch

import audiocraft

from . import builders
from .encodec import CompressionModel


def get_audiocraft_cache_dir() -> tp.Optional[str]:
    return os.environ.get('AUDIOCRAFT_CACHE_DIR', None)


HF_MODEL_CHECKPOINTS_MAP = {
    "small": "facebook/musicgen-small",
    "medium": "facebook/musicgen-medium",
    "large": "facebook/musicgen-large",
    "melody": "facebook/musicgen-melody",
    "melody-large": "facebook/musicgen-melody-large",
    "stereo-small": "facebook/musicgen-stereo-small",
    "stereo-medium": "facebook/musicgen-stereo-medium",
    "stereo-large": "facebook/musicgen-stereo-large",
    "stereo-melody": "facebook/musicgen-stereo-melody",
    "stereo-melody-large": "facebook/musicgen-stereo-melody-large",
    "style": "facebook/musicgen-style",
}


def _get_state_dict(
    file_or_url_or_id: tp.Union[Path, str],
    filename: tp.Optional[str] = None,
    device='cpu',
    cache_dir: tp.Optional[str] = None,
):
    if cache_dir is None:
        cache_dir = get_audiocraft_cache_dir()
    # Return the state dict either from a file or url
    file_or_url_or_id = str(file_or_url_or_id)
    assert isinstance(file_or_url_or_id, str)

    if os.path.isfile(file_or_url_or_id):
        return torch.load(file_or_url_or_id, map_location=device)
    
    if os.path.isdir(file_or_url_or_id):
        file = f"{file_or_url_or_id}/{filename}"
        return torch.load(file, map_location=device)

    elif file_or_url_or_id.startswith('https://'):
        return torch.hub.load_state_dict_from_url(file_or_url_or_id, map_location=device, check_hash=True)

    elif file_or_url_or_id in HF_MODEL_CHECKPOINTS_MAP:
        assert filename is not None, "filename needs to be defined if using HF checkpoints"

        repo_id = HF_MODEL_CHECKPOINTS_MAP[file_or_url_or_id]
        file = hf_hub_download(repo_id=repo_id, filename=filename, cache_dir=cache_dir)
        return torch.load(file, map_location=device)

    else:
        assert filename is not None, "filename needs to be defined if using HF checkpoints"

        file = hf_hub_download(
            repo_id=file_or_url_or_id, filename=filename, cache_dir=cache_dir,
            library_name="audiocraft", library_version=audiocraft.__version__)
        return torch.load(file, map_location=device)

def create_melody_config(model_id: str, device: str) -> DictConfig:
    """Create a fallback configuration for melody models.
    
    Args:
        model_id: The model identifier
        device: The device to use
    
    Returns:
        A compatible OmegaConf DictConfig
    """
    base_cfg = {
        "device": str(device),
        "channels": 2 if "stereo" in model_id else 1,
        "sample_rate": 32000,
        "audio_channels": 2 if "stereo" in model_id else 1,
        "frame_rate": 50,
        "codec_name": "encodec",
        "codec": {
            "dim": 128,
            "hidden_dim": 1024,
            "stride": 320,
            "n_q": 4,
            "codebook_size": 2048,
            "normalize": True,
        }
    }
    return OmegaConf.create(base_cfg)

def create_default_config(model_id: str, device: str) -> DictConfig:
    """Create a fallback configuration for standard models.
    
    Args:
        model_id: The model identifier
        device: The device to use
    
    Returns:
        A compatible OmegaConf DictConfig
    """
    base_cfg = {
        "device": str(device),
        "channels": 2 if "stereo" in model_id else 1,
        "sample_rate": 32000,
        "audio_channels": 2 if "stereo" in model_id else 1,
        "frame_rate": 50,
        "codec_name": "encodec",
        "codec": {
            "dim": 128,
            "hidden_dim": 1024,
            "stride": 320,
            "n_q": 4,
            "codebook_size": 1024,
            "normalize": True,
        }
    }
    return OmegaConf.create(base_cfg)


def load_compression_model_ckpt(file_or_url_or_id: tp.Union[Path, str], cache_dir: tp.Optional[str] = None):
    return _get_state_dict(file_or_url_or_id, filename="compression_state_dict.bin", cache_dir=cache_dir)


def load_compression_model(file_or_url_or_id: tp.Union[Path, str], device='cpu', cache_dir: tp.Optional[str] = None):
    pkg = load_compression_model_ckpt(file_or_url_or_id, cache_dir=cache_dir)
    if 'pretrained' in pkg:
        return CompressionModel.get_pretrained(pkg['pretrained'], device=device)
    
    # Handle newer model formats that might not have xp.cfg
    if 'xp.cfg' not in pkg:
        if file_or_url_or_id in ['melody-large', 'stereo-melody', 'stereo-medium', 
                                 'stereo-small', 'stereo-large', 'stereo-melody-large','style']:
            print(f"Using fallback configuration for {file_or_url_or_id}")
            # Create a default configuration based on the model type
            # This is where you'd need to add model-specific configurations
            if 'melody' in file_or_url_or_id:
                cfg = create_melody_config(file_or_url_or_id, device)
            else:
                cfg = create_default_config(file_or_url_or_id, device)
        else:
            raise KeyError(f"Missing configuration for model {file_or_url_or_id}")
    else:
        cfg = OmegaConf.create(pkg['xp.cfg'])
    
    cfg.device = str(device)
    model = builders.get_compression_model(cfg)
    model.load_state_dict(pkg['best_state'])
    model.eval()
    return model

def load_lm_model_ckpt(file_or_url_or_id: tp.Union[Path, str], cache_dir: tp.Optional[str] = None):
    return _get_state_dict(file_or_url_or_id, filename="state_dict.bin", cache_dir=cache_dir)


def _delete_param(cfg: DictConfig, full_name: str):
    parts = full_name.split('.')
    for part in parts[:-1]:
        if part in cfg:
            cfg = cfg[part]
        else:
            return
    OmegaConf.set_struct(cfg, False)
    if parts[-1] in cfg:
        del cfg[parts[-1]]
    OmegaConf.set_struct(cfg, True)


def load_lm_model(file_or_url_or_id: tp.Union[Path, str], device='cpu', cache_dir: tp.Optional[str] = None):
    pkg = load_lm_model_ckpt(file_or_url_or_id, cache_dir=cache_dir)
    cfg = OmegaConf.create(pkg['xp.cfg'])
    cfg.device = str(device)
    if cfg.device == 'cpu':
        cfg.transformer_lm.memory_efficient = False
        cfg.transformer_lm.custom = True
        cfg.dtype = 'float32'
    else:
        cfg.dtype = 'float16'
    _delete_param(cfg, 'conditioners.self_wav.chroma_stem.cache_path')
    _delete_param(cfg, 'conditioners.args.merge_text_conditions_p')
    _delete_param(cfg, 'conditioners.args.drop_desc_p')
    model = builders.get_lm_model(cfg)
    model.load_state_dict(pkg['best_state'])
    model.eval()
    model.cfg = cfg
    return model


def load_lm_model_magnet(file_or_url_or_id: tp.Union[Path, str], compression_model_frame_rate: int,
                         device='cpu', cache_dir: tp.Optional[str] = None):
    pkg = load_lm_model_ckpt(file_or_url_or_id, cache_dir=cache_dir)
    cfg = OmegaConf.create(pkg['xp.cfg'])
    cfg.device = str(device)
    if cfg.device == 'cpu':
        cfg.dtype = 'float32'
    else:
        cfg.dtype = 'float16'
    _delete_param(cfg, 'conditioners.args.merge_text_conditions_p')
    _delete_param(cfg, 'conditioners.args.drop_desc_p')

    cfg.transformer_lm.compression_model_framerate = compression_model_frame_rate
    cfg.transformer_lm.segment_duration = cfg.dataset.segment_duration
    cfg.transformer_lm.span_len = cfg.masking.span_len

    # MAGNeT models v1 support only xformers backend.
    from audiocraft.modules.transformer import set_efficient_attention_backend

    if cfg.transformer_lm.memory_efficient:
        set_efficient_attention_backend("xformers")

    model = builders.get_lm_model(cfg)
    model.load_state_dict(pkg['best_state'])
    model.eval()
    model.cfg = cfg
    return model


def load_jasco_model(file_or_url_or_id: tp.Union[Path, str],
                     compression_model: CompressionModel,
                     device='cpu', cache_dir: tp.Optional[str] = None):
    pkg = load_lm_model_ckpt(file_or_url_or_id, cache_dir=cache_dir)
    cfg = OmegaConf.create(pkg['xp.cfg'])
    cfg.device = str(device)
    if cfg.device == 'cpu':
        cfg.dtype = 'float32'
    else:
        cfg.dtype = 'float16'
    model = builders.get_jasco_model(cfg, compression_model)
    model.load_state_dict(pkg['best_state'])
    model.eval()
    model.cfg = cfg
    return model


def load_mbd_ckpt(file_or_url_or_id: tp.Union[Path, str],
                  filename: tp.Optional[str] = None,
                  cache_dir: tp.Optional[str] = None):
    return _get_state_dict(file_or_url_or_id, filename=filename, cache_dir=cache_dir)


def load_diffusion_models(file_or_url_or_id: tp.Union[Path, str],
                          device='cpu',
                          filename: tp.Optional[str] = None,
                          cache_dir: tp.Optional[str] = None):
    pkg = load_mbd_ckpt(file_or_url_or_id, filename=filename, cache_dir=cache_dir)
    models = []
    processors = []
    cfgs = []
    sample_rate = pkg['sample_rate']
    for i in range(pkg['n_bands']):
        cfg = pkg[i]['cfg']
        model = builders.get_diffusion_model(cfg)
        model_dict = pkg[i]['model_state']
        model.load_state_dict(model_dict)
        model.to(device)
        processor = builders.get_processor(cfg=cfg.processor, sample_rate=sample_rate)
        processor_dict = pkg[i]['processor_state']
        processor.load_state_dict(processor_dict)
        processor.to(device)
        models.append(model)
        processors.append(processor)
        cfgs.append(cfg)
    return models, processors, cfgs