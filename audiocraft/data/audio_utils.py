# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
"""Various utilities for audio convertion (pcm format, sample rate and channels),
and volume normalization."""
import sys
import typing as tp

import julius
import torch
import torchaudio


def convert_audio_channels(wav: torch.Tensor, channels: int = 2) -> torch.Tensor:
    """Convert audio to the given number of channels.

    Args:
        wav (torch.Tensor): Audio wave of shape [B, C, T].
        channels (int): Expected number of channels as output.
    Returns:
        torch.Tensor: Downmixed or unchanged audio wave [B, C, T].
    """
    *shape, src_channels, length = wav.shape
    if src_channels == channels:
        pass
    elif channels == 1:
        # Case 1:
        # The caller asked 1-channel audio, and the stream has multiple
        # channels, downmix all channels.
        wav = wav.mean(dim=-2, keepdim=True)
    elif src_channels == 1:
        # Case 2:
        # The caller asked for multiple channels, but the input file has
        # a single channel, replicate the audio over all channels.
        wav = wav.expand(*shape, channels, length)
    elif src_channels >= channels:
        # Case 3:
        # The caller asked for multiple channels, and the input file has
        # more channels than requested. In that case return the first channels.
        wav = wav[..., :channels, :]
    else:
        # Case 4: What is a reasonable choice here?
        raise ValueError('The audio file has less channels than requested but is not mono.')
    return wav


def convert_audio(wav: torch.Tensor, from_rate: float,
                  to_rate: float, to_channels: int) -> torch.Tensor:
    """Convert audio to new sample rate and number of audio channels.
    """
    wav = julius.resample_frac(wav, int(from_rate), int(to_rate))
    wav = convert_audio_channels(wav, to_channels)
    return wav


def normalize_loudness(wav: torch.Tensor, sample_rate: int, loudness_headroom_db: float = 14,
                       loudness_compressor: bool = False, energy_floor: float = 2e-3):
    """Normalize an input signal to a user loudness in dB LKFS.
    Audio loudness is defined according to the ITU-R BS.1770-4 recommendation.

    Args:
        wav (torch.Tensor): Input multichannel audio data.
        sample_rate (int): Sample rate.
        loudness_headroom_db (float): Target loudness of the output in dB LUFS.
        loudness_compressor (bool): Uses tanh for soft clipping.
        energy_floor (float): anything below that RMS level will not be rescaled.
    Returns:
        output (torch.Tensor): Loudness normalized output data.
    """
    energy = wav.pow(2).mean().sqrt().item()
    if energy < energy_floor:
        return wav
    transform = torchaudio.transforms.Loudness(sample_rate)
    input_loudness_db = transform(wav).item()
    # calculate the gain needed to scale to the desired loudness level
    delta_loudness = -loudness_headroom_db - input_loudness_db
    gain = 10.0 ** (delta_loudness / 20.0)
    output = gain * wav
    if loudness_compressor:
        output = torch.tanh(output)
    assert output.isfinite().all(), (input_loudness_db, wav.pow(2).mean().sqrt())
    return output


def _clip_wav(wav: torch.Tensor, log_clipping: bool = False, stem_name: tp.Optional[str] = None) -> None:
    """Utility function to clip the audio with logging if specified."""
    max_scale = wav.abs().max()
    if log_clipping and max_scale > 1:
        clamp_prob = (wav.abs() > 1).float().mean().item()
        print(f"CLIPPING {stem_name or ''} happening with proba (a bit of clipping is okay):",
              clamp_prob, "maximum scale: ", max_scale.item(), file=sys.stderr)
    wav.clamp_(-1, 1)


def normalize_audio(wav: torch.Tensor, normalize: bool = True,
                    strategy: str = 'peak', peak_clip_headroom_db: float = 1,
                    rms_headroom_db: float = 18, loudness_headroom_db: float = 14,
                    loudness_compressor: bool = False, log_clipping: bool = False,
                    sample_rate: tp.Optional[int] = None,
                    stem_name: tp.Optional[str] = None) -> torch.Tensor:
    """Normalize the audio according to the prescribed strategy (see after).

    Args:
        wav (torch.Tensor): Audio data.
        normalize (bool): if `True` (default), normalizes according to the prescribed
            strategy (see after). If `False`, the strategy is only used in case clipping
            would happen.
        strategy (str): Can be either 'clip', 'peak', or 'rms'. Default is 'peak',
            i.e. audio is normalized by its largest value. RMS normalizes by root-mean-square
            with extra headroom to avoid clipping. 'clip' just clips.
        peak_clip_headroom_db (float): Headroom in dB when doing 'peak' or 'clip' strategy.
        rms_headroom_db (float): Headroom in dB when doing 'rms' strategy. This must be much larger
            than the `peak_clip` one to avoid further clipping.
        loudness_headroom_db (float): Target loudness for loudness normalization.
        loudness_compressor (bool): If True, uses tanh based soft clipping.
        log_clipping (bool): If True, basic logging on stderr when clipping still
            occurs despite strategy (only for 'rms').
        sample_rate (int): Sample rate for the audio data (required for loudness).
        stem_name (Optional[str]): Stem name for clipping logging.
    Returns:
        torch.Tensor: Normalized audio.
    """
    scale_peak = 10 ** (-peak_clip_headroom_db / 20)
    scale_rms = 10 ** (-rms_headroom_db / 20)
    if strategy == 'peak':
        rescaling = (scale_peak / wav.abs().max())
        if normalize or rescaling < 1:
            wav = wav * rescaling
    elif strategy == 'clip':
        wav = wav.clamp(-scale_peak, scale_peak)
    elif strategy == 'rms':
        mono = wav.mean(dim=0)
        rescaling = scale_rms / mono.pow(2).mean().sqrt()
        if normalize or rescaling < 1:
            wav = wav * rescaling
        _clip_wav(wav, log_clipping=log_clipping, stem_name=stem_name)
    elif strategy == 'loudness':
        assert sample_rate is not None, "Loudness normalization requires sample rate."
        wav = normalize_loudness(wav, sample_rate, loudness_headroom_db, loudness_compressor)
        _clip_wav(wav, log_clipping=log_clipping, stem_name=stem_name)
    else:
        assert wav.abs().max() < 1
        assert strategy == '' or strategy == 'none', f"Unexpected strategy: '{strategy}'"
    return wav


def f32_pcm(wav: torch.Tensor) -> torch.Tensor:
    """Convert audio to float 32 bits PCM format.
    """
    if wav.dtype.is_floating_point:
        return wav
    elif wav.dtype == torch.int16:
        return wav.float() / 2**15
    elif wav.dtype == torch.int32:
        return wav.float() / 2**31
    raise ValueError(f"Unsupported wav dtype: {wav.dtype}")


def i16_pcm(wav: torch.Tensor) -> torch.Tensor:
    """Convert audio to int 16 bits PCM format.

    ..Warning:: There exist many formula for doing this conversion. None are perfect
    due to the asymmetry of the int16 range. One either have possible clipping, DC offset,
    or inconsistencies with f32_pcm. If the given wav doesn't have enough headroom,
    it is possible that `i16_pcm(f32_pcm)) != Identity`.
    """
    if wav.dtype.is_floating_point:
        assert wav.abs().max() <= 1
        candidate = (wav * 2 ** 15).round()
        if candidate.max() >= 2 ** 15:  # clipping would occur
            candidate = (wav * (2 ** 15 - 1)).round()
        return candidate.short()
    else:
        assert wav.dtype == torch.int16
        return wav

def apply_tafade(audio: torch.Tensor, sample_rate, duration=3.0, out=True, start=True, shape: str = "linear", stem_name: tp.Optional[str] = None) -> torch.Tensor:
    """
    Apply fade-in and/or fade-out effects to the audio tensor.

    Args:
        audio (torch.Tensor): The input audio tensor of shape (C, L).
        sample_rate (int): The sample rate of the audio.
        duration (float, optional): The duration of the fade in seconds. Defaults to 3.0.
        out (bool, optional): Determines whether to apply fade-in (False) or fade-out (True) effect. Defaults to True.
        start (bool, optional): Determines whether the fade is applied to the beginning (True) or end (False) of the audio. Defaults to True.
        shape (str, optional): The shape of the fade. Must be one of: "quarter_sine", "half_sine", "linear", "logarithmic", "exponential". Defaults to "linear".

    Returns:
        torch.Tensor: The audio tensor with the fade effect applied.

    """
    fade_samples = int(sample_rate * duration)  # Number of samples for the fade duration

    # Create the fade transform
    fade_transform = torchaudio.transforms.Fade(fade_in_len=0, fade_out_len=0, fade_shape=shape)

    if out:
        fade_transform.fade_out_len = fade_samples
    else:
        fade_transform.fade_in_len = fade_samples        

    # Select the portion of the audio to apply the fade
    if start:
        audio_fade_section = audio[:, :fade_samples]
    else:
        audio_fade_section = audio[:, -fade_samples:]

    # Apply the fade transform to the audio section
    audio_faded = fade_transform(audio)

    # Replace the selected portion of the audio with the faded section
    if start:
        audio_faded[:, :fade_samples] = audio_fade_section
    else:
        audio_faded[:, -fade_samples:] = audio_fade_section

    wav = normalize_loudness(audio_faded,sample_rate, loudness_headroom_db=18, loudness_compressor=True)
    _clip_wav(wav, log_clipping=False, stem_name=stem_name)
    return wav


def apply_fade(audio: torch.Tensor, sample_rate, duration=3.0, out=True, start=True, curve_start:float=0.0, curve_end:float=1.0, current_device:str="cpu", stem_name: tp.Optional[str] = None) -> torch.Tensor:
    """
    Apply fade-in and/or fade-out effects to the audio tensor.

    Args:
        audio (torch.Tensor): The input audio tensor of shape (C, L).
        sample_rate (int): The sample rate of the audio.
        duration (float, optional): The duration of the fade in seconds. Defaults to 3.0.
        out (bool, optional): Determines whether to apply fade-in (False) or fade-out (True) effect. Defaults to True.
        start (bool, optional): Determines whether the fade is applied to the beginning (True) or end (False) of the audio. Defaults to True.
        curve_start (float, optional): The starting amplitude of the fade curve. Defaults to 0.0.
        curve_end (float, optional): The ending amplitude of the fade curve. Defaults to 1.0.
        current_device (str, optional): The device on which the fade curve tensor should be created. Defaults to "cpu".

    Returns:
        torch.Tensor: The audio tensor with the fade effect applied.

    """
    fade_samples = int(sample_rate * duration)  # Number of samples for the fade duration
    fade_curve = torch.linspace(curve_start, curve_end, fade_samples, device=current_device)  # Generate linear fade curve

    if out:
        fade_curve = fade_curve.flip(0)  # Reverse the fade curve for fade out

    # Select the portion of the audio to apply the fade
    if start:
        audio_fade_section = audio[:, :fade_samples]
    else:
        audio_fade_section = audio[:, -fade_samples:]

    # Apply the fade curve to the audio section
    audio_faded = audio.clone()
    audio_faded[:, :fade_samples] *= fade_curve.unsqueeze(0)
    audio_faded[:, -fade_samples:] *= fade_curve.unsqueeze(0)

    # Replace the selected portion of the audio with the faded section
    if start:
        audio_faded[:, :fade_samples] = audio_fade_section
    else:
        audio_faded[:, -fade_samples:] = audio_fade_section

    wav = normalize_loudness(audio_faded,sample_rate, loudness_headroom_db=18, loudness_compressor=True)
    _clip_wav(wav, log_clipping=False, stem_name=stem_name)
    return wav

def apply_splice_effect(waveform1, sample_rate1, waveform2, sample_rate2, overlap):
    # Convert sample rates to integers
    sample_rate1 = int(sample_rate1)
    sample_rate2 = int(sample_rate2)

    # Convert tensors to mono-channel if needed
    if waveform1.ndim > 2:
        waveform1 = waveform1.mean(dim=1)
    if waveform2.ndim > 2:
        waveform2 = waveform2.mean(dim=1)

    ## Convert tensors to numpy arrays
    #waveform1_np = waveform1.numpy()
    #waveform2_np = waveform2.numpy()

    # Apply splice effect using torchaudio.sox_effects.apply_effects_tensor
    effects = [
        ["splice", f"-q {waveform1},{overlap}"],
    ]
    output_waveform, output_sample_rate = torchaudio.sox_effects.apply_effects_tensor(
        torch.cat([waveform1.unsqueeze(0), waveform2.unsqueeze(0)], dim=2),
        sample_rate1,
        effects
    )

    return output_waveform.squeeze(0), output_sample_rate

