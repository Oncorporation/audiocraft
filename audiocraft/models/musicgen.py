# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
Main model for using MusicGen. This will combine all the required components
and provide easy access to the generation API.
"""

import os
import typing as tp
import warnings

import omegaconf
import torch
import gradio as gr

from .encodec import CompressionModel
from .genmodel import BaseGenModel
from .lm import LMModel
from .builders import get_debug_compression_model, get_debug_lm_model, get_wrapped_compression_model
from .loaders import load_compression_model, load_lm_model, HF_MODEL_CHECKPOINTS_MAP
from ..data.audio_utils import convert_audio
from ..modules.conditioners import ConditioningAttributes, WavCondition, StyleConditioner
from ..utils.autocast import TorchAutocast

MelodyList = tp.List[tp.Optional[torch.Tensor]]
MelodyType = tp.Union[torch.Tensor, MelodyList]


class MusicGen:
    """MusicGen main model with convenient generation API.

    Args:
        name (str): name of the model.
        compression_model (CompressionModel): Compression model
            used to map audio to invertible discrete representations.
        lm (LMModel): Language model over discrete representations.
        max_duration (float, optional): maximum duration the model can produce,
            otherwise, inferred from the training params.
    """
    def __init__(self, name: str, compression_model: CompressionModel, lm: LMModel, max_duration: tp.Optional[float] = 30):
        self.name = name
        self.compression_model = compression_model
        self.lm = lm
        self.cfg: tp.Optional[omegaconf.DictConfig] = None
        # Just to be safe, let's put everything in eval mode.
        self.compression_model.eval()
        self.lm.eval()

        if hasattr(lm, 'cfg'):
            cfg = lm.cfg
            assert isinstance(cfg, omegaconf.DictConfig)
            self.cfg = cfg

        if self.cfg is not None:
            self.compression_model = get_wrapped_compression_model(self.compression_model, self.cfg)

        if max_duration is None:
            if self.cfg is not None:
                max_duration = lm.cfg.dataset.segment_duration  # type: ignore
            else:
                raise ValueError("You must provide max_duration when building directly MusicGen")
        assert max_duration is not None
        self.max_duration = max_duration
        self.duration = 15.0  # default duration
        self.device = next(iter(lm.parameters())).device
        self.generation_params: dict = {}
        self.set_generation_params(duration=self.duration)  # 15 seconds by default
        self._progress_callback: tp.Union[tp.Callable[[int, int], None], gr.Progress] = None
        if self.device.type == 'cpu':
            self.autocast = TorchAutocast(enabled=False)
        else:
            self.autocast = TorchAutocast(
                enabled=True, device_type=self.device.type, dtype=torch.float16)

    @property
    def version(self) -> str:
        from audiocraft import __version__ as audiocraft_version
        return audiocraft_version

    @property
    def frame_rate(self) -> float:
        """Roughly the number of AR steps per seconds."""
        return self.compression_model.frame_rate

    @property
    def sample_rate(self) -> int:
        """Sample rate of the generated audio."""
        return self.compression_model.sample_rate

    @property
    def audio_channels(self) -> int:
        """Audio channels of the generated audio."""
        return self.compression_model.channels

    @staticmethod
    def get_pretrained(name: str = 'melody-large', device=None):
        """Return pretrained model, we provide ten models:
        - small (300M), text to music, # see: https://huggingface.co/facebook/musicgen-small
        - medium (1.5B), text to music, # see: https://huggingface.co/facebook/musicgen-medium
        - melody (1.5B) text to music and text+melody to music, # see: https://huggingface.co/facebook/musicgen-melody
        - large (3.3B), text to music, # see: https://huggingface.co/facebook/musicgen-large
        - melody-large (3.3B), text to music, and text+melody to music # see: https://huggingface.co/facebook/musicgen-melody-large
        - stereo-small (300M), text to music, # see: https://huggingface.co/facebook/musicgen-small
        - stereo-medium (1.5B), text to music, # see: https://huggingface.co/facebook/musicgen-stereo-medium
        - stereo-melody (1.5B) text to music and text+melody to music, # see: https://huggingface.co/facebook/musicgen-stereo-melody
        - stereo-large (3.3B), text to music, # see: https://huggingface.co/facebook/musicgen-stereo-large
        - stereo-melody-large (3.3B), text to music, and text+melody to music # see: https://huggingface.co/facebook/musicgen-stereo-melody-large
        - musicgen-style (1.5B), text to music, # see: https://huggingface.co/facebook/musicgen-style
        """

        if device is None:
            if torch.cuda.device_count():
                device = 'cuda'
            else:
                device = 'cpu'

        if name == 'debug':
            # used only for unit tests
            compression_model = get_debug_compression_model(device)
            lm = get_debug_lm_model(device)
            return MusicGen(name, compression_model, lm, max_duration=30)

        if name not in HF_MODEL_CHECKPOINTS_MAP:
            if not os.path.isfile(name) and not os.path.isdir(name):
                raise ValueError(
                    f"{name} is not a valid checkpoint name. "
                    f"Choose one of {', '.join(HF_MODEL_CHECKPOINTS_MAP.keys())}"
                )
        else:
            name = HF_MODEL_CHECKPOINTS_MAP[name]

        cache_dir = os.environ.get('MUSICGEN_ROOT', None)
        compression_model = load_compression_model(name, device=device, cache_dir=cache_dir)
        lm = load_lm_model(name, device=device, cache_dir=cache_dir)
        if name.__contains__('melody') or 'self_wav' in lm.condition_provider.conditioners:
            lm.condition_provider.conditioners['self_wav'].match_len_on_eval = True
            lm.condition_provider.conditioners['self_wav']._use_masking = False

        return MusicGen(name, compression_model, lm)

    def set_generation_params(self, use_sampling: bool = True, top_k: int = 250,
                              top_p: float = 0.0, temperature: float = 1.0,
                              duration: float = 30.0, cfg_coef: float = 3.0,
                              cfg_coef_beta: tp.Optional[float] = None,
                              two_step_cfg: bool = False, extend_stride: float = 10, rep_penalty: float = None):
        """Set the generation parameters for MusicGen.

        Args:
            use_sampling (bool, optional): Use sampling if True, else do argmax decoding. Defaults to True.
            top_k (int, optional): top_k used for sampling. Defaults to 250.
            top_p (float, optional): top_p used for sampling, when set to 0 top_k is used. Defaults to 0.0.
            temperature (float, optional): Softmax temperature parameter. Defaults to 1.0.
            duration (float, optional): Duration of the generated waveform. Defaults to 30.0.
            cfg_coef (float, optional): Coefficient used for classifier free guidance. Defaults to 3.0.
            cfg_coef_beta (float, optional): beta coefficient in double classifier free guidance.
                Should be only used for MusicGen melody if we want to push the text condition more than
                the audio conditioning. See paragraph 4.3 in https://arxiv.org/pdf/2407.12563 to understand
                double CFG.
            two_step_cfg (bool, optional): If True, performs 2 forward for Classifier Free Guidance,
                instead of batching together the two. This has some impact on how things
                are padded but seems to have little impact in practice.
            extend_stride: when doing extended generation (i.e. more than 30 seconds), by how much
                should we extend the audio each time. Larger values will mean less context is
                preserved, and shorter value will require extra computations.
            rep_penalty (float, optional): If set, use repetition penalty during generation. Not Implemented.
        """
        assert extend_stride < self.max_duration, "Cannot stride by more than max generation duration."
        self.extend_stride = extend_stride
        self.duration = duration
        self.generation_params = {
            #'max_gen_len': int(duration * self.frame_rate),
            'use_sampling': use_sampling,
            'temp': temperature,
            'top_k': top_k,
            'top_p': top_p,
            'cfg_coef': cfg_coef,
            'two_step_cfg': two_step_cfg,
            'cfg_coef_beta': cfg_coef_beta,
        }

    def set_style_conditioner_params(self, eval_q: int = 3, excerpt_length: float = 3.0,
                                     ds_factor: tp.Optional[int] = None,
                                     encodec_n_q: tp.Optional[int] = None) -> None:
        """Set the parameters of the style conditioner
        Args:
            eval_q (int): the number of residual quantization streams used to quantize the style condition
                the smaller it is, the narrower is the information bottleneck
            excerpt_length (float): the excerpt length in seconds that is extracted from the audio
                conditioning
            ds_factor: (int): the downsampling factor used to downsample the style tokens before
                using them as a prefix
            encodec_n_q: (int, optional): if encodec is used as a feature extractor, sets the number
                of streams that is used to extract features
        """
        assert isinstance(self.lm.condition_provider.conditioners.self_wav, StyleConditioner), \
            "Only use this function if you model is MusicGen-Style"
        self.lm.condition_provider.conditioners.self_wav.set_params(eval_q=eval_q,
                                                                    excerpt_length=excerpt_length,
                                                                    ds_factor=ds_factor,
                                                                    encodec_n_q=encodec_n_q)

    def set_custom_progress_callback(self, progress_callback: tp.Union[tp.Callable[[int, int], None],gr.Progress] = None):
        """Override the default progress callback."""
        self._progress_callback = progress_callback

    def generate_unconditional(self, num_samples: int, progress: bool = False,
                               return_tokens: bool = False, progress_callback: gr.Progress = None) -> tp.Union[torch.Tensor, 
                                                                        tp.Tuple[torch.Tensor, torch.Tensor]]:
        """Generate samples in an unconditional manner.

        Args:
            num_samples (int): Number of samples to be generated.
            progress (bool, optional): Flag to display progress of the generation process. Defaults to False.
            return_tokens (bool, optional): If True, also return the generated tokens. Defaults to False.
        """
        descriptions: tp.List[tp.Optional[str]] = [None] * num_samples
        attributes, prompt_tokens = self._prepare_tokens_and_attributes(descriptions, None)
        tokens = self._generate_tokens(attributes, prompt_tokens, progress)
        if return_tokens:
            return self.generate_audio(tokens), tokens
        return self.generate_audio(tokens)

    def generate(self, descriptions: tp.List[str], progress: bool = False, return_tokens: bool = False, progress_callback: gr.Progress = None) \
            -> tp.Union[torch.Tensor, tp.Tuple[torch.Tensor, torch.Tensor]]:
        """Generate samples conditioned on text.

        Args:
            descriptions (list of str): A list of strings used as text conditioning.
            progress (bool, optional): Flag to display progress of the generation process. Defaults to False.
            return_tokens (bool, optional): If True, also return the generated tokens. Defaults to False.
        """
        attributes, prompt_tokens = self._prepare_tokens_and_attributes(descriptions, None)
        assert prompt_tokens is None
        tokens = self._generate_tokens(attributes, prompt_tokens, progress)
        if return_tokens:
            return self.generate_audio(tokens), tokens
        return self.generate_audio(tokens)

    def generate_with_chroma(self, descriptions: tp.List[str], melody_wavs: MelodyType,
                             melody_sample_rate: int, progress: bool = False,
                             return_tokens: bool = False, progress_callback=gr.Progress(track_tqdm=True)) -> tp.Union[torch.Tensor,
                                                                      tp.Tuple[torch.Tensor, torch.Tensor]]:
        """Generate samples conditioned on text and melody.

        Args:
            descriptions (list of str): A list of strings used as text conditioning.
            melody_wavs: (torch.Tensor or list of Tensor): A batch of waveforms used as
                melody conditioning. Should have shape [B, C, T] with B matching the description length,
                C=1 or 2. It can be [C, T] if there is a single description. It can also be
                a list of [C, T] tensors.
            melody_sample_rate: (int): Sample rate of the melody waveforms.
            progress (bool, optional): Flag to display progress of the generation process. Defaults to False.
            return_tokens (bool, optional): If True, also return the generated tokens. Defaults to False.
        """
        if isinstance(melody_wavs, torch.Tensor):
            if melody_wavs.dim() == 2:
                melody_wavs = melody_wavs[None]
            if melody_wavs.dim() != 3:
                raise ValueError("Melody wavs should have a shape [B, C, T].")
            melody_wavs = list(melody_wavs)
        else:
            for melody in melody_wavs:
                if melody is not None:
                    assert melody.dim() == 2, "One melody in the list has the wrong number of dims."

        melody_wavs = [
            convert_audio(wav, melody_sample_rate, self.sample_rate, self.audio_channels)
            if wav is not None else None
            for wav in melody_wavs]
        attributes, prompt_tokens = self._prepare_tokens_and_attributes(descriptions=descriptions, prompt=None,
                                                                        melody_wavs=melody_wavs)
        assert prompt_tokens is None
        tokens = self._generate_tokens(attributes, prompt_tokens, progress)
        if return_tokens:
            return self.generate_audio(tokens), tokens
        return self.generate_audio(tokens)

    def generate_with_all(self, descriptions: tp.List[str], melody_wavs: MelodyType,
                             sample_rate: int, progress: bool = False, prompt: tp.Optional[torch.Tensor] = None, return_tokens: bool = False, progress_callback: gr.Progress = None) \
            -> tp.Union[torch.Tensor, tp.Tuple[torch.Tensor, torch.Tensor]]:
        """Generate samples conditioned on text and melody and audio prompts.
        Args:
            descriptions (tp.List[str]): A list of strings used as text conditioning.
            melody_wavs: (torch.Tensor or list of Tensor): A batch of waveforms used as
                melody conditioning. Should have shape [B, C, T] with B matching the description length,
                C=1 or 2. It can be [C, T] if there is a single description. It can also be
                a list of [C, T] tensors.
           sample_rate: (int): Sample rate of the melody waveforms.
           progress (bool, optional): Flag to display progress of the generation process. Defaults to False.
           prompt (torch.Tensor): A batch of waveforms used for continuation.
                Prompt should be [B, C, T], or [C, T] if only one sample is generated.
        """
        if isinstance(melody_wavs, torch.Tensor):
            if melody_wavs.dim() == 2:
                melody_wavs = melody_wavs[None]
            if melody_wavs.dim() != 3:
                raise ValueError("Melody wavs should have a shape [B, C, T].")
            melody_wavs = list(melody_wavs)
        else:
            for melody in melody_wavs:
                if melody is not None:
                    assert melody.dim() == 2, "One melody in the list has the wrong number of dims."

        melody_wavs = [
            convert_audio(wav, sample_rate, self.sample_rate, self.audio_channels)
            if wav is not None else None
            for wav in melody_wavs]
        #attributes, prompt_tokens = self._prepare_tokens_and_attributes(descriptions=descriptions, prompt=None,
        #                                                                melody_wavs=melody_wavs)

        if prompt is not None:
            if prompt.dim() == 2:
                prompt = prompt[None]
            if prompt.dim() != 3:
                raise ValueError("prompt should have 3 dimensions: [B, C, T] (C = 1).")
            prompt = convert_audio(prompt, sample_rate, self.sample_rate, self.audio_channels)
        if descriptions is None:
            descriptions = [None] * len(prompt)

        #if prompt is not None:
        #    attributes_gen, prompt_tokens = self._prepare_tokens_and_attributes(descriptions, prompt)
        
        attributes, prompt_tokens = self._prepare_tokens_and_attributes(descriptions=descriptions, prompt=prompt,
                                                                        melody_wavs=melody_wavs)
        if prompt is not None:
            assert prompt_tokens is not None
        else:
            assert prompt_tokens is None
        tokens = self._generate_tokens(attributes, prompt_tokens, progress)
        if return_tokens:
            return self.generate_audio(tokens), tokens
        return self.generate_audio(tokens)

    def generate_continuation(self, prompt: torch.Tensor, prompt_sample_rate: int,
                              descriptions: tp.Optional[tp.List[tp.Optional[str]]] = None,
                              progress: bool = False, return_tokens: bool = False, progress_callback: gr.Progress = None) \
            -> tp.Union[torch.Tensor, tp.Tuple[torch.Tensor, torch.Tensor]]:
        """Generate samples conditioned on audio prompts.

        Args:
            prompt (torch.Tensor): A batch of waveforms used for continuation.
                Prompt should be [B, C, T], or [C, T] if only one sample is generated.
            prompt_sample_rate (int): Sampling rate of the given audio waveforms.
            descriptions (list of str, optional): A list of strings used as text conditioning. Defaults to None.
            progress (bool, optional): Flag to display progress of the generation process. Defaults to False.
            return_tokens (bool, optional): If True, also return the generated tokens. Defaults to False.\
            This is truly a hack and does not follow the progression of conditioning melody or previously generated audio.
        """
        if prompt.dim() == 2:
            prompt = prompt[None]
        if prompt.dim() != 3:
            raise ValueError("prompt should have 3 dimensions: [B, C, T] (C = 1).")
        prompt = convert_audio(prompt, prompt_sample_rate, self.sample_rate, self.audio_channels)
        if descriptions is None:
            descriptions = [None] * len(prompt)
        attributes, prompt_tokens = self._prepare_tokens_and_attributes(descriptions, prompt)
        assert prompt_tokens is not None
        tokens = self._generate_tokens(attributes, prompt_tokens, progress)
        if return_tokens:
            return self.generate_audio(tokens), tokens
        return self.generate_audio(tokens)

    @torch.no_grad()
    def _prepare_tokens_and_attributes(
            self,
            descriptions: tp.Sequence[tp.Optional[str]],
            prompt: tp.Optional[torch.Tensor],
            melody_wavs: tp.Optional[MelodyList] = None, 
            progress_callback: tp.Optional[gr.Progress] = None
    ) -> tp.Tuple[tp.List[ConditioningAttributes], tp.Optional[torch.Tensor]]:
        """Prepare model inputs.

        Args:
            descriptions (list of str): A list of strings used as text conditioning.
            prompt (torch.Tensor): A batch of waveforms used for continuation.
            melody_wavs (torch.Tensor, optional): A batch of waveforms
                used as melody conditioning. Defaults to None.
        """
        attributes = [
            ConditioningAttributes(text={'description': description})
            for description in descriptions]

        if melody_wavs is None:
            for attr in attributes:
                attr.wav['self_wav'] = WavCondition(
                    torch.zeros((1, 1, 1), device=self.device),
                    torch.tensor([0], device=self.device),
                    sample_rate=[self.sample_rate],
                    path=[None]) # type: ignore
        else:
            if 'self_wav' not in self.lm.condition_provider.conditioners:
                raise RuntimeError("This model doesn't support melody conditioning. "
                                   "Use the `melody` model.")
            assert len(melody_wavs) == len(descriptions), \
                f"number of melody wavs must match number of descriptions! " \
                f"got melody len={len(melody_wavs)}, and descriptions len={len(descriptions)}"
            for attr, melody in zip(attributes, melody_wavs):
                if melody is None:
                    attr.wav['self_wav'] = WavCondition(
                        torch.zeros((1, 1, 1), device=self.device),
                        torch.tensor([0], device=self.device),
                        sample_rate=[self.sample_rate],
                        path=[None]) # type: ignore
                else:
                    attr.wav['self_wav'] = WavCondition(
                        melody[None].to(device=self.device),
                        torch.tensor([melody.shape[-1]], device=self.device),
                        sample_rate=[self.sample_rate],
                        path=[None],
                    )

        if prompt is not None:
            if descriptions is not None:
                assert len(descriptions) == len(prompt), "Prompt and nb. descriptions doesn't match"
            prompt = prompt.to(self.device)
            prompt_tokens, scale = self.compression_model.encode(prompt)
            assert scale is None
        else:
            prompt_tokens = None
        return attributes, prompt_tokens

    def _generate_tokens(self, attributes: tp.List[ConditioningAttributes],
                         prompt_tokens: tp.Optional[torch.Tensor], progress: bool = False, progress_callback: gr.Progress = None) -> torch.Tensor:
        """Generate discrete audio tokens given audio prompt and/or conditions.

        Args:
            attributes (list of ConditioningAttributes): Conditions used for generation (text/melody).
            prompt_tokens (torch.Tensor, optional): Audio prompt used for continuation.
            progress (bool, optional): Flag to display progress of the generation process. Defaults to False.
        Returns:
            torch.Tensor: Generated audio, of shape [B, C, T], T is defined by the generation params.
        """
        total_gen_len = int(self.duration * self.frame_rate)
        max_prompt_len = int(min(self.duration, self.max_duration) * self.frame_rate)
        current_gen_offset: int = 0

        def _progress_callback(generated_tokens: int, tokens_to_generate: int):
            generated_tokens += current_gen_offset
            generated_tokens /= ((tokens_to_generate) / self.duration)
            tokens_to_generate /= ((tokens_to_generate) / self.duration)
            if self._progress_callback is not None:
                # Note that total_gen_len might be quite wrong depending on the
                # codebook pattern used, but with delay it is almost accurate.
                self._progress_callback((generated_tokens / tokens_to_generate), f"Generated {generated_tokens: 6.2f}/{tokens_to_generate: 6.2f} seconds")
            if progress_callback is not None:
                # Update Gradio progress bar
                progress_callback((generated_tokens / tokens_to_generate), f"Generated {generated_tokens: 6.2f}/{tokens_to_generate: 6.2f} seconds")
            if progress:
                print(f'{generated_tokens: 6.2f} / {tokens_to_generate: 6.2f}', end='\r')

        if prompt_tokens is not None:
            if prompt_tokens.shape[-1] > max_prompt_len:
                prompt_tokens = prompt_tokens[..., :max_prompt_len]

        # callback = None
        callback = _progress_callback

        if self.duration <= self.max_duration:
            # generate by sampling from LM, simple case.
            with self.autocast:
                gen_tokens = self.lm.generate(
                    prompt_tokens, attributes,
                    callback=callback, max_gen_len=total_gen_len, **self.generation_params)

        else:
            # now this gets a bit messier, we need to handle prompts,
            # melody conditioning etc.
            ref_wavs = [attr.wav['self_wav'] for attr in attributes]
            all_tokens = []
            if prompt_tokens is None:
                prompt_length = 0
            else:
                all_tokens.append(prompt_tokens)
                prompt_length = prompt_tokens.shape[-1]

            stride_tokens = int(self.frame_rate * self.extend_stride)

            while current_gen_offset + prompt_length < total_gen_len:
                time_offset = current_gen_offset / self.frame_rate
                chunk_duration = min(self.duration - time_offset, self.max_duration)
                max_gen_len = int(chunk_duration * self.frame_rate)
                for attr, ref_wav in zip(attributes, ref_wavs):
                    wav_length = ref_wav.length.item()
                    if wav_length == 0:
                        continue
                    # We will extend the wav periodically if it not long enough.
                    # we have to do it here rather than in conditioners.py as otherwise
                    # we wouldn't have the full wav.
                    initial_position = int(time_offset * self.sample_rate)
                    wav_target_length = int(self.max_duration * self.sample_rate)
                    print(initial_position / self.sample_rate, wav_target_length / self.sample_rate)
                    positions = torch.arange(initial_position,
                                             initial_position + wav_target_length, device=self.device)
                    attr.wav['self_wav'] = WavCondition(
                        ref_wav[0][..., positions % wav_length],
                        torch.full_like(ref_wav[1], wav_target_length),
                        [self.sample_rate] * ref_wav[0].size(0),
                        [None], [0.])
                with self.autocast:
                    gen_tokens = self.lm.generate(
                        prompt_tokens, attributes,
                        callback=callback, max_gen_len=max_gen_len, **self.generation_params)
                if prompt_tokens is None:
                    all_tokens.append(gen_tokens)
                else:
                    all_tokens.append(gen_tokens[:, :, prompt_tokens.shape[-1]:])
                prompt_tokens = gen_tokens[:, :, stride_tokens:]
                prompt_length = prompt_tokens.shape[-1]
                current_gen_offset += stride_tokens

            gen_tokens = torch.cat(all_tokens, dim=-1)
        return gen_tokens

        # generate audio

    def generate_audio(self, gen_tokens: torch.Tensor):
        try:
            """Generate Audio from tokens"""
            assert gen_tokens.dim() == 3
            with torch.no_grad():
                gen_audio = self.compression_model.decode(gen_tokens, None)
            return gen_audio
        except Exception as e:
            print(f"Error generating audio: {e}")
            return None
    
    #def _generate_tokens(self, attributes: tp.List[ConditioningAttributes],
    #                     prompt_tokens: tp.Optional[torch.Tensor], progress: bool = False) -> torch.Tensor:
    #    """Generate discrete audio tokens given audio prompt and/or conditions.

    #    Args:
    #        attributes (tp.List[ConditioningAttributes]): Conditions used for generation (text/melody).
    #        prompt_tokens (tp.Optional[torch.Tensor]): Audio prompt used for continuation.
    #        progress (bool, optional): Flag to display progress of the generation process. Defaults to False.
    #    Returns:
    #        torch.Tensor: Generated audio, of shape [B, C, T], T is defined by the generation params.
    #    """
    #    def _progress_callback(generated_tokens: int, tokens_to_generate: int):
    #        print(f'{generated_tokens: 6d} / {tokens_to_generate: 6d}', end='\r')

    #    if prompt_tokens is not None:
    #        assert self.generation_params['max_gen_len'] > prompt_tokens.shape[-1], \
    #            "Prompt is longer than audio to generate"

    #    callback = None
    #    if progress:
    #        callback = _progress_callback

    #    # generate by sampling from LM
    #    with self.autocast:
    #        gen_tokens = self.lm.generate(prompt_tokens, attributes, callback=callback, **self.generation_params)

    #    # generate audio
    #    assert gen_tokens.dim() == 3
    #    with torch.no_grad():
    #        gen_audio = self.compression_model.decode(gen_tokens, None)
    #    return gen_audio

    def to(self, device: str):
        self.compression_model.to(device)
        self.lm.to(device)
        return self