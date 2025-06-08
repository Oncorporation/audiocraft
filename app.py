"""
Copyright (c) Meta Platforms, Inc. and affiliates.
All rights reserved.

This source code is licensed under the license found in the
LICENSE file in the root directory of this source tree.
"""

from tempfile import NamedTemporaryFile
import argparse
import torch
import gradio as gr
import os
import subprocess
import sys
from pathlib import Path
import time
import typing as tp
import warnings
import gc
from tqdm import tqdm
from audiocraft.models import MusicGen
from audiocraft.data.audio import audio_write
from audiocraft.data.audio_utils import apply_fade, apply_tafade, apply_splice_effect
from audiocraft.utils.extend import generate_music_segments, add_settings_to_image, INTERRUPTING
from audiocraft.utils import utils
import numpy as np
import random
import shutil
from mutagen.mp4 import MP4
#from typing import List, Union
import librosa
import modules.user_history
from modules.version_info import versions_html, commit_hash, get_xformers_version
from modules.gradio import *
from modules.file_utils import get_file_parts, get_filename_from_filepath, convert_title_to_filename, get_unique_file_path, delete_file

MODEL = None
MODELS = None
IS_SHARED_SPACE = "Surn/UnlimitedMusicGen" in os.environ.get('SPACE_ID', '')
INTERRUPTED = False
UNLOAD_MODEL = False
MOVE_TO_CPU = False
MAX_PROMPT_INDEX = 0
git = os.environ.get('GIT', "git")
#s.environ["CUDA_LAUNCH_BLOCKING"] = "1"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128,expandable_segments:True"
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['CUDA_MODULE_LOADING']='LAZY'
os.environ['USE_FLASH_ATTENTION'] = '1'
os.environ['XFORMERS_FORCE_DISABLE_TRITON']= '1'

def interrupt_callback():
    return INTERRUPTED

def interrupt():
    global INTERRUPTING
    INTERRUPTING = True

class FileCleaner:
    def __init__(self, file_lifetime: float = 3600):
        self.file_lifetime = file_lifetime
        self.files = []

    def add(self, path: tp.Union[str, Path]):
        self._cleanup()
        self.files.append((time.time(), Path(path)))

    def _cleanup(self):
        now = time.time()
        for time_added, path in list(self.files):
            if now - time_added > self.file_lifetime:
                if path.exists():
                    path.unlink()
                self.files.pop(0)
            else:
                break


#file_cleaner = FileCleaner()

def toggle_audio_src(choice):
    if choice == "mic":
        return gr.update(source="microphone", value=None, label="Microphone")
    else:
        return gr.update(source="upload", value=None, label="File")

def get_waveform(*args, **kwargs):
    # Further remove some warnings.
    be = time.time()
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        out = gr.make_waveform(*args, **kwargs)
        print("Make a video took", time.time() - be)
        return out
        

def load_model(version, progress=gr.Progress(track_tqdm=True)):
    global MODEL, MODELS, UNLOAD_MODEL
    print("Loading model", version)

    with tqdm(total=100, desc=f"Loading model '{version}'", unit="step") as pbar:
        if MODELS is None:
            pbar.update(50)  # Simulate progress for loading
            result = MusicGen.get_pretrained(version)
            pbar.update(50)  # Complete progress
            return result
        else:
            t1 = time.monotonic()
            if MODEL is not None:
                MODEL.to('cpu')  # Move to cache
                print("Previous model moved to CPU in %.2fs" % (time.monotonic() - t1))
                pbar.update(30)  # Simulate progress for moving model to CPU
                t1 = time.monotonic()
            if MODELS.get(version) is None:
                print("Loading model %s from disk" % version)
                result = MusicGen.get_pretrained(version)
                MODELS[version] = result
                print("Model loaded in %.2fs" % (time.monotonic() - t1))
                pbar.update(70)  # Simulate progress for loading from disk
                return result
            result = MODELS[version].to('cuda')
            print("Cached model loaded in %.2fs" % (time.monotonic() - t1))
            pbar.update(100)  # Complete progress
            return result

def get_melody(melody_filepath):
        audio_data= list(librosa.load(melody_filepath, sr=None))
        audio_data[0], audio_data[1] = audio_data[1], audio_data[0]
        melody = tuple(audio_data)
        return melody

def git_tag():
    try:
        return subprocess.check_output([git, "describe", "--tags"], shell=False, encoding='utf8').strip()
    except Exception:
        try:
            from pathlib import Path
            changelog_md = Path(__file__).parent.parent / "CHANGELOG.md"
            with changelog_md.open(encoding="utf-8") as file:
                return next((line.strip() for line in file if line.strip()), "<none>")
        except Exception:
            return "<none>"

def load_background_filepath(video_orientation):
    if video_orientation == "Landscape":
        return "./assets/background.png"
    else:
        return "./assets/background_portrait.png"

def load_melody_filepath(melody_filepath, title, assigned_model,topp, temperature, cfg_coef, segment_length = 30):
    # get melody filename
    #$Union[str, os.PathLike]
    symbols = ['_', '.', '-']
    MAX_OVERLAP = int(segment_length  // 2) - 1
    if (melody_filepath is None) or (melody_filepath == ""):
        return title, gr.update(maximum=0, value=-1) , gr.update(value="medium", interactive=True), gr.update(value=topp), gr.update(value=temperature), gr.update(value=cfg_coef), gr.update(maximum=MAX_OVERLAP)
    
    if (title is None) or ("MusicGen" in title) or (title == ""):
        melody_name, melody_extension = get_filename_from_filepath(melody_filepath)
        # fix melody name for symbols
        for symbol in symbols:
            melody_name = melody_name.replace(symbol, ' ').title()
        #additonal melody setting updates
        topp = 800
        temperature = 0.5
        cfg_coef = 3.25
    else:
        melody_name = title

    if ("melody" not in assigned_model):
        assigned_model = "melody-large"

    print(f"Melody name: {melody_name}, Melody Filepath: {melody_filepath}, Model: {assigned_model}\n")

    # get melody length in number of segments and modify the UI
    melody = get_melody(melody_filepath)
    sr, melody_data = melody[0], melody[1]
    segment_samples = sr * segment_length
    total_melodys = max(min((len(melody_data) // segment_samples), 25), 0) 
    print(f"Melody length: {len(melody_data)}, Melody segments: {total_melodys}\n")
    MAX_PROMPT_INDEX = total_melodys
    
    return  gr.update(value=melody_name), gr.update(maximum=MAX_PROMPT_INDEX, value=-1), gr.update(value=assigned_model, interactive=True), gr.update(value=topp), gr.update(value=temperature), gr.update(value=cfg_coef), gr.update(maximum=MAX_OVERLAP)

def predict(model, text, melody_filepath, duration, dimension, topk, topp, temperature, cfg_coef, background, title, settings_font, settings_font_color, seed, overlap=1, prompt_index = 0, include_title = True, include_settings = True, harmony_only = False, profile = gr.OAuthProfile, segment_length = 30, settings_font_size=28, settings_animate_waveform=False, video_orientation="Landscape", excerpt_duration=3.5, progress=gr.Progress(track_tqdm=True)):
    global MODEL, INTERRUPTED, INTERRUPTING, MOVE_TO_CPU
    output_segments = None
    melody_name = "Not Used"
    melody_extension = "Not Used"
    melody = None
    if melody_filepath:
        melody_name, melody_extension = get_filename_from_filepath(melody_filepath)
        melody = get_melody(melody_filepath)

    INTERRUPTED = False
    INTERRUPTING = False
    if temperature < 0:
        temperature = 0.1
        raise gr.Error("Temperature must be >= 0.")
    if topk < 0:
        topk = 1
        raise gr.Error("Topk must be non-negative.")
    if topp < 0:
        topp =1
        raise gr.Error("Topp must be non-negative.")

    # Clean up GPU resources only if the model changes
    if MODEL is not None and model not in MODEL.name:
        print(f"Switching model from {MODEL.name} to {model}. Cleaning up resources.")
        del MODEL  # Delete the current model
        torch.cuda.empty_cache()  # Clear GPU memory
        gc.collect()  # Force garbage collection
        MODEL = None

    try:
        if MODEL is None or model not in MODEL.name:
            MODEL = load_model(model)
        else:
            if MOVE_TO_CPU:
                MODEL.to('cuda')
    except Exception as e:
        raise gr.Error(f"Error loading model '{model}': {str(e)}. Try a different model.")
    
    # prevent hacking
    duration = min(duration, 720)
    overlap =  min(overlap, 15)
    # 

    output = None
    segment_duration = duration
    initial_duration = duration
    output_segments = []
    while duration > 0:
        if not output_segments: # first pass of long or short song
            if segment_duration > MODEL.lm.cfg.dataset.segment_duration:
                segment_duration = MODEL.lm.cfg.dataset.segment_duration
            else:
                segment_duration = duration
        else: # next pass of long song
            if duration + overlap < MODEL.lm.cfg.dataset.segment_duration:
                segment_duration = duration + overlap
            else:
                segment_duration = MODEL.lm.cfg.dataset.segment_duration
        if (segment_length + overlap) < segment_duration:
            segment_duration = segment_length + overlap
        # implement seed
        if seed < 0:
            seed = random.randint(0, 0xffff_ffff_ffff)
        torch.manual_seed(seed)


        print(f'Segment duration: {segment_duration}, duration: {duration}, overlap: {overlap}')
        if ("style" in model) and melody:
            # style and text-to-music
            MODEL.set_generation_params(
                use_sampling=True,
                top_k=topk,
                top_p=topp,
                temperature=temperature,
                cfg_coef=cfg_coef,
                duration=segment_duration,
                two_step_cfg=False,
                cfg_coef_beta=5, # double CFG is only useful for text-and-style conditioning
            )

            MODEL.set_style_conditioner_params(
                eval_q=3, # integer between 1 and 6
                            # eval_q is the level of quantization that passes
                            # through the conditioner. When low, the models adheres less to the 
                            # audio conditioning
                excerpt_length=excerpt_duration, # the length in seconds that is taken by the model in the provided excerpt, can be
                                    # between 1.5 and 4.5 seconds but it has to be shortest to the length of the provided conditioning
            )
        else:
            MODEL.set_generation_params(
                use_sampling=True,
                top_k=topk,
                top_p=topp,
                temperature=temperature,
                cfg_coef=cfg_coef,
                duration=segment_duration,
                two_step_cfg=False,
                extend_stride=2,
                rep_penalty=0.5,
                cfg_coef_beta=None, # double CFG is only useful for text-and-style conditioning
            )
        MODEL.set_custom_progress_callback(gr.Progress(track_tqdm=True))

        try:
            if melody and ("melody" or "style" in model):
                # return excess duration, load next model and continue in loop structure building up output_segments
                if duration > MODEL.duration:
                    output_segments, duration = generate_music_segments(text, melody, seed, MODEL, duration, overlap, MODEL.duration, prompt_index, harmony_only, excerpt_duration, progress=gr.Progress(track_tqdm=True))
                else:
                    # pure original code
                    sr, melody = melody[0], torch.from_numpy(melody[1]).to(MODEL.device).float().t().unsqueeze(0)
                    print(melody.shape)
                    if melody.dim() == 2:
                        melody = melody[None]
                    melody = melody[..., :int(sr * MODEL.lm.cfg.dataset.segment_duration)]
                    output = MODEL.generate_with_chroma(
                        descriptions=[text],
                        melody_wavs=melody,
                        melody_sample_rate=sr,
                        progress=False, progress_callback=gr.Progress(track_tqdm=True)
                    )
                # All output_segments are populated, so we can break the loop or set duration to 0
                break
            else:
                #output = MODEL.generate(descriptions=[text], progress=False)
                if not output_segments:
                    next_segment = MODEL.generate(descriptions=[text], progress=False, progress_callback=gr.Progress(track_tqdm=True))
                    duration -= segment_duration
                else:
                    last_chunk = output_segments[-1][:, :, -overlap*MODEL.sample_rate:]
                    next_segment = MODEL.generate_continuation(last_chunk, MODEL.sample_rate, descriptions=[text], progress=False, progress_callback=gr.Progress(track_tqdm=True))
                    duration -= segment_duration - overlap
                if next_segment != None:
                    output_segments.append(next_segment)
        except Exception as e:
            print(f"Error generating audio: {e}")
            gr.Error(f"Error generating audio: {e}")
            return None, None, seed

        if INTERRUPTING:
            INTERRUPTED = True
            INTERRUPTING = False
            print("Function execution interrupted!")
            raise gr.Error("Interrupted.")

    print(f"\nOutput segments: {len(output_segments)}\n")
    if output_segments:
        try:
            # Combine the output segments into one long audio file or stack tracks
            #output_segments = [segment.detach().cpu().float()[0] for segment in output_segments]
            #output = torch.cat(output_segments, dim=dimension)
            
            output = output_segments[0]
            for i in range(1, len(output_segments)):
                if overlap > 0:
                    overlap_samples = overlap * MODEL.sample_rate
                    #stack tracks and fade out/in
                    overlapping_output_fadeout = output[:, :, -overlap_samples:]
                    #overlapping_output_fadeout = apply_fade(overlapping_output_fadeout,sample_rate=MODEL.sample_rate,duration=overlap,out=True,start=True, curve_end=0.0, current_device=MODEL.device)
                    overlapping_output_fadeout = apply_tafade(overlapping_output_fadeout,sample_rate=MODEL.sample_rate,duration=overlap,out=True,start=True,shape="linear")

                    overlapping_output_fadein = output_segments[i][:, :, :overlap_samples]
                    #overlapping_output_fadein = apply_fade(overlapping_output_fadein,sample_rate=MODEL.sample_rate,duration=overlap,out=False,start=False, curve_start=0.0, current_device=MODEL.device)
                    overlapping_output_fadein = apply_tafade(overlapping_output_fadein,sample_rate=MODEL.sample_rate,duration=overlap,out=False,start=False, shape="linear")
                    
                    overlapping_output = torch.cat([overlapping_output_fadeout[:, :, :-(overlap_samples // 2)], overlapping_output_fadein],dim=2)
                    ###overlapping_output, overlap_sample_rate = apply_splice_effect(overlapping_output_fadeout, MODEL.sample_rate, overlapping_output_fadein, MODEL.sample_rate, overlap)
                    print(f" overlap size Fade:{overlapping_output.size()}\n output: {output.size()}\n segment: {output_segments[i].size()}")
                    ##overlapping_output = torch.cat([output[:, :, -overlap_samples:], output_segments[i][:, :, :overlap_samples]], dim=1) #stack tracks
                    ##print(f" overlap size stack:{overlapping_output.size()}\n output: {output.size()}\n segment: {output_segments[i].size()}")
                    #overlapping_output = torch.cat([output[:, :, -overlap_samples:], output_segments[i][:, :, :overlap_samples]], dim=2) #stack tracks
                    #print(f" overlap size cat:{overlapping_output.size()}\n output: {output.size()}\n segment: {output_segments[i].size()}")
                    output = torch.cat([output[:, :, :-overlap_samples], overlapping_output, output_segments[i][:, :, overlap_samples:]], dim=dimension)
                else:
                    output = torch.cat([output, output_segments[i]], dim=dimension)
            output = output.detach().cpu().float()[0]
        except Exception as e:
            print(f"Error combining segments: {e}. Using the first segment only.")
            output = output_segments[0].detach().cpu().float()[0]
    else:
        if (output is None) or (output.dim() == 0):
            return None, None, seed
        else:
            output = output.detach().cpu().float()[0]
    
    video_width, video_height = 768, 512
    if video_orientation == "Portait":
        video_width, video_height = 512, 768

    title_file_name = convert_title_to_filename(title)
    with NamedTemporaryFile("wb", suffix=".wav", delete=False, prefix=title_file_name) as file:
        video_description = f"{text}\n Duration: {str(initial_duration)} Dimension: {dimension}\n Top-k:{topk} Top-p:{topp}\n Randomness:{temperature}\n cfg:{cfg_coef} overlap: {overlap}\n Seed: {seed}\n Model: {model}\n Melody Condition:{melody_name}\n Sample Segment: {prompt_index}"
        if include_settings or include_title:
            background = add_settings_to_image(title if include_title else "",video_description if include_settings else "",width=video_width, height=video_height, background_path=background,font=settings_font,font_color=settings_font_color, font_size=settings_font_size)
        audio_write(
            file.name, output, MODEL.sample_rate, strategy="loudness",
            loudness_headroom_db=18, loudness_compressor=True, add_suffix=False, channels=2)
        waveform_video_path = get_waveform(file.name, bg_image=background, bar_count=45, name=title_file_name, animate=settings_animate_waveform, progress=gr.Progress(track_tqdm=True))
        # Remove the extension from file.name
        file_name_without_extension = os.path.splitext(file.name)[0]
        # Get the directory, filename, name, extension, and new extension of the waveform video path
        video_dir, video_name, video_name, video_ext, video_new_ext = get_file_parts(waveform_video_path)

        new_video_path = get_unique_file_path(video_dir, title_file_name, video_new_ext)

        mp4 = MP4(waveform_video_path)
        mp4["©nam"] = title_file_name        # Title tag
        mp4["desc"] = f"{text}\n Duration: {str(initial_duration)}" # Description tag

        commit = commit_hash()
        metadata = {
            "Title": title,
            "Year": time.strftime("%Y"),
            "prompt": text,
            "negative_prompt": "",
            "Seed": seed,
            "steps": 1,
            "wdth": video_width,
            "hght": video_height,
            "Dimension": dimension,
            "Top-k": topk,
            "Top-p": topp,
            "Randomness": temperature,
            "cfg": cfg_coef,
            "overlap": overlap,
            "Melody Condition": melody_name,
            "Sample Segment": prompt_index,
            "Duration": initial_duration,
            "Audio": file.name,
            "font": settings_font,
            "font_color": settings_font_color,
            "font_size": settings_font_size,
            "harmony_only": harmony_only,
            "background": background,
            "include_title": include_title,
            "include_settings": include_settings,
            "profile": "Satoshi Nakamoto" if profile.value is None else profile.value.username,
            "commit": commit_hash(),
            "tag": git_tag(),
            "version": gr.__version__,
            "model_version": MODEL.version,
            "model_name": MODEL.name,
            "model_description": f"{MODEL.audio_channels} channels, {MODEL.sample_rate} Hz",
            "melody_name": melody_name if melody_name else "",
            "melody_extension": melody_extension if melody_extension else "",
            "hostname": "https://huggingface.co/spaces/Surn/UnlimitedMusicGen",
            "version": f"https://huggingface.co/spaces/Surn/UnlimitedMusicGen/commit/{'huggingface' if commit == '<none>' else commit}",
            "python": sys.version,
            "torch": getattr(torch, '__long_version__', torch.__version__),
            "xformers": get_xformers_version(),
            "gradio": gr.__version__,
            "huggingface_space": os.environ.get('SPACE_ID', ''),
            "CUDA": f"{'CUDA is available. device: ' + torch.cuda.get_device_name(0) + ' version: ' + torch.version.cuda if torch.cuda.is_available() else 'CUDA is not available.'}",
        }
        # Add additional metadata from the metadata dictionary (if it exists)
        for key, value in metadata.items():
            mp4[key] = str(value)  # Convert values to strings as required by mutagen

        # Save the metadata changes to the file
        mp4.save()
        
        try:
            os.replace(waveform_video_path, new_video_path)
            waveform_video_path = new_video_path
        except Exception as e:
            print(f"Error renaming file: {e}")

        if waveform_video_path:
            modules.user_history.save_file(
            profile=profile.value,
            image=background,
            audio=file.name,
            video=waveform_video_path,
            label=title,
            metadata=metadata,
            progress=gr.Progress(track_tqdm=True)
        )
        
        
    if MOVE_TO_CPU:
        MODEL.to('cpu')
    if UNLOAD_MODEL:
        MODEL = None

    # Explicitly delete large tensors or objects
    del output_segments, output, melody, melody_name, melody_extension, metadata, mp4
    
    # Force garbage collection
    #gc.collect()
    
    # Synchronize CUDA streams
    torch.cuda.synchronize()

    #torch.cuda.empty_cache()
    torch.cuda.ipc_collect()
    return waveform_video_path, file.name, seed

gr.set_static_paths(paths=["fonts/","assets/", "images/"])
def ui(**kwargs):
    with gr.Blocks(title="UnlimitedMusicGen",css_paths="style_20250331.css", theme='Surn/beeuty') as demo:
        with gr.Tab("UnlimitedMusicGen"):
            gr.Markdown(
                """
            # UnlimitedMusicGen
            This is your private demo for [UnlimitedMusicGen](https://github.com/Oncorporation/audiocraft), a simple and controllable model for music generation
            presented at: ["Simple and Controllable Music Generation"](https://huggingface.co/papers/2306.05284)
            
            Disclaimer: This won't run on CPU only. Clone this App and run on GPU instance!
                        
            Todo: Working on improved Interrupt.
            Theme Available at ["Surn/Beeuty"](https://huggingface.co/spaces/Surn/Beeuty)

                """
            )
            if IS_SHARED_SPACE and not torch.cuda.is_available():
                gr.Markdown("""
                    ⚠ This Space doesn't work in this shared UI ⚠

                        <a href="https://huggingface.co/spaces/musicgen/MusicGen?duplicate=true" style="display: inline-block;margin-top: .5em;margin-right: .25em;" target="_blank">
                        <img style="margin-bottom: 0em;display: inline;margin-top: -.25em;" src="https://bit.ly/3gLdBN6" alt="Duplicate Space"></a>
                        to use it privately, or use the <a href="https://huggingface.co/spaces/facebook/MusicGen">public demo</a>
                        """)
            with gr.Row():
                    with gr.Column():
                        with gr.Row():
                            with gr.Column():
                                text = gr.Text(label="Describe your music", interactive=True, value="4/4 100bpm 320kbps 32khz, Industrial/Electronic Soundtrack, Dark, Intense, Sci-Fi, soft fade-in, soft fade-out", key="prompt", lines=4)
                                autoplay_cb = gr.Checkbox(value=False, label="Autoplay?", key="autoplay_cb")
                            with gr.Column():
                                duration = gr.Slider(minimum=1, maximum=720, value=10, label="Duration (s)", interactive=True, key="total_duration", step=1)
                                model = gr.Radio(["melody", "medium", "small", "large", "melody-large", "stereo-small", "stereo-medium", "stereo-large", "stereo-melody", "stereo-melody-large", "style"], label="AI Model", value="medium", interactive=True, key="chosen_model")
                        with gr.Row():
                            submit = gr.Button("Generate", elem_id="btn-generate")
                            # Adapted from https://github.com/rkfg/audiocraft/blob/long/app.py, MIT license.
                            _ = gr.Button("Interrupt", elem_id="btn-interrupt").click(fn=interrupt, queue=False)
                        with gr.Row():
                            with gr.Column():
                                radio = gr.Radio(["file", "mic"], value="file", label="Condition on a melody (optional) File or Mic")
                                melody_filepath = gr.Audio(sources=["upload"], type="filepath", label="Melody Condition (optional)", interactive=True, elem_id="melody-input", key="melody_input")
                            with gr.Column():
                                harmony_only = gr.Radio(label="Use Harmony Only",choices=["No", "Yes"], value="No", interactive=True, info="Remove Drums?", key="use_harmony")
                                prompt_index = gr.Slider(label="Melody Condition Sample Segment", minimum=-1, maximum=MAX_PROMPT_INDEX, step=1, value=-1, interactive=True, info="Which 10-30 second segment to condition with, - 1  = align with conditioning melody", key="melody_index")
                        with gr.Accordion("Video", open=False):
                            with gr.Row():
                                background= gr.Image(value="./assets/background.png", sources=["upload"], label="Background", width=768, height=512, type="filepath", interactive=True, key="background_imagepath")
                                with gr.Column():
                                    include_title = gr.Checkbox(label="Add Title", value=True, interactive=True,key="add_title")
                                    include_settings = gr.Checkbox(label="Add Settings to background", value=True, interactive=True, key="add_settings")
                                    video_orientation = gr.Radio(label="Video Orientation", choices=["Landscape", "Portait"], value="Landscape", interactive=True, key="video_orientation")
                            with gr.Row():
                                title = gr.Textbox(label="Title", value="UnlimitedMusicGen", interactive=True, key="song_title")
                                settings_font = gr.Text(label="Settings Font", value="./assets/arial.ttf", interactive=True)
                                settings_font_color = gr.ColorPicker(label="Settings Font Color", value="#c87f05", interactive=True, key="settings_font_color")
                                settings_font_size = gr.Slider(minimum=8, maximum=64, value=28, step=1, label="Settings Font Size", interactive=True, key="settings_font_size")
                                settings_animate_waveform = gr.Checkbox(label="Animate Waveform", value=False, interactive=True, key="animate_waveform")
                        with gr.Accordion("Expert", open=False):
                            with gr.Row():
                                segment_length = gr.Slider(minimum=10, maximum=30, value=30, step=1,label="Music Generation Segment Length (s)", interactive=True,key="segment_length")
                                overlap = gr.Slider(minimum=0, maximum=14, value=1, step=1, label="Segment Overlap", interactive=True)
                                dimension = gr.Slider(minimum=-2, maximum=2, value=2, step=1, label="Dimension", info="determines which direction to add new segements of audio. (1 = stack tracks, 2 = lengthen, -2..0 = ?)", interactive=True)
                            with gr.Row():
                                topk = gr.Number(label="Top-k", value=280, precision=0, interactive=True, info="more structured", key="topk")
                                topp = gr.Number(label="Top-p", value=1150, precision=0, interactive=True, info="more variation, overwrites Top-k if not zero", key="topp")
                                temperature = gr.Number(label="Randomness Temperature", value=0.7, precision=None, step=0.1, interactive=True, info="less than one to follow Melody Condition song closely", key="temperature")
                                cfg_coef = gr.Number(label="Classifier Free Guidance", value=3.75, precision=None, step=0.1, interactive=True, info="3.0-4.0, stereo and small need more", key="cfg_coef")
                            with gr.Row():
                                seed = gr.Number(label="Seed", value=-1, precision=0, interactive=True, key="seed")
                                gr.Button('\U0001f3b2\ufe0f', elem_classes="small-btn").click(fn=lambda: -1, outputs=[seed], queue=False)
                                reuse_seed = gr.Button('\u267b\ufe0f', elem_classes="small-btn")
                    with gr.Column() as c:
                        output = gr.Video(label="Generated Music", interactive=False, show_download_button=True, show_share_button=True, autoplay=False)
                        wave_file = gr.File(label=".wav file", elem_id="output_wavefile", interactive=True)
                        seed_used = gr.Number(label='Seed used', value=-1, interactive=False)

            radio.change(toggle_audio_src, radio, [melody_filepath], queue=False, show_progress=False, api_name="audio_src_change")
            video_orientation.change(load_background_filepath, inputs=[video_orientation], outputs=[background], queue=False, show_progress=False, api_name="video_orientation_change")
            melody_filepath.change(load_melody_filepath, inputs=[melody_filepath, title, model,topp, temperature, cfg_coef, segment_length], outputs=[title, prompt_index , model, topp, temperature, cfg_coef, overlap], api_name="melody_filepath_change", queue=False)
            reuse_seed.click(fn=lambda x: x, inputs=[seed_used], outputs=[seed], queue=False, api_name="reuse_seed_click")
            autoplay_cb.change(fn=lambda x: gr.update(autoplay=x), inputs=[autoplay_cb], outputs=[output], queue=False, api_name="autoplay_cb_change")
            segment_length.release(fn=load_melody_filepath, queue=False, api_name="segment_length_change", trigger_mode="once", inputs=[melody_filepath, title, model,topp, temperature, cfg_coef, segment_length], outputs=[title, prompt_index , model, topp, temperature, cfg_coef, overlap], show_progress="minimal")

            gr.Examples(
                examples=[
                    [
                        "4/4 120bpm 320kbps 32khz, An 80s driving pop song with heavy drums and synth pads in the background",
                        "./assets/bach.mp3",
                        "melody",
                        "80s Pop Synth",
                        950,
                        0.6,
                        3.5
                    ],
                    [
                        "4/4 120bpm 320kbps 32khz, A cheerful country song with acoustic guitars",
                        "./assets/bolero_ravel.mp3",
                        "stereo-melody-large",
                        "Country Guitar",
                        750,
                        0.7,
                        4.0
                    ],
                    [
                        "4/4 120bpm 320kbps 32khz, 90s rock song with electric guitar and heavy drums",
                        None,
                        "stereo-medium", 
                        "90s Rock Guitar",
                        1150,
                        0.7,
                        3.75
                    ],
                    [
                        "4/4 120bpm 320kbps 32khz, a light and cheery EDM track, with syncopated drums, aery pads, and strong emotions",
                        "./assets/bach.mp3",
                        "melody-large",
                        "EDM my Bach",
                        500,
                        0.7,
                        3.75
                    ],
                    [
                        "4/4 320kbps 32khz, lofi slow bpm electro chill with organic samples",
                        None,
                        "medium", 
                        "LoFi Chill",
                        0,
                        0.7,
                        4.0
                    ],
                ],
                inputs=[text, melody_filepath, model, title, topp, temperature, cfg_coef],
                outputs=[output]
            )
            
        with gr.Tab("User History") as history_tab:
            modules.user_history.setup(display_type="video_path")
            modules.user_history.render()
            
        user_profile = gr.State(None)
            
        with gr.Row("Versions") as versions_row:
            gr.HTML(value=versions_html(), visible=True, elem_id="versions")

        submit.click(
            modules.user_history.get_profile,
            inputs=[],
            outputs=[user_profile],
            queue=True,
            api_name="submit"
         ).then(
             predict,
             inputs=[model, text,melody_filepath, duration, dimension, topk, topp, temperature, cfg_coef, background, title, settings_font, settings_font_color, seed, overlap, prompt_index, include_title, include_settings, harmony_only, user_profile, segment_length, settings_font_size, settings_animate_waveform, video_orientation],
             outputs=[output, wave_file, seed_used], scroll_to_output=True)

        # Show the interface
        launch_kwargs = {}
        username = kwargs.get('username')
        password = kwargs.get('password')
        server_port = kwargs.get('server_port', 0)
        inbrowser = kwargs.get('inbrowser', False)
        share = kwargs.get('share', False)
        server_name = kwargs.get('listen')

        launch_kwargs['server_name'] = server_name

        if username and password:
            launch_kwargs['auth'] = (username, password)
        if server_port > 0:
            launch_kwargs['server_port'] = server_port
        if inbrowser:
            launch_kwargs['inbrowser'] = inbrowser
        if share:
            launch_kwargs['share'] = share
        launch_kwargs['allowed_paths'] = ["assets", "./assets", "images", "./images", 'e:/TMP']
        launch_kwargs['favicon_path'] = "./assets/favicon.ico"
        launch_kwargs['mcp_server'] = True


        demo.queue(max_size=10, api_open=True).launch(**launch_kwargs)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--listen',
        type=str,
        default='0.0.0.0' if 'SPACE_ID' in os.environ else '127.0.0.1',
        help='IP to listen on for connections to Gradio',
    )
    parser.add_argument(
        '--username', type=str, default='', help='Username for authentication'
    )
    parser.add_argument(
        '--password', type=str, default='', help='Password for authentication'
    )
    parser.add_argument(
        '--server_port',
        type=int,
        default=7859,
        help='Port to run the server listener on',
    )
    parser.add_argument(
        '--inbrowser', action='store_true', help='Open in browser'
    )
    parser.add_argument(
        '--share', action='store_true', help='Share the gradio UI'
    )
    parser.add_argument(
        '--unload_model', action='store_true', help='Unload the model after every generation to save GPU memory'
    )

    parser.add_argument(
        '--unload_to_cpu', action='store_true', help='Move the model to main RAM after every generation to save GPU memory but reload faster than after full unload (see above)'
    )

    parser.add_argument(
        '--cache', action='store_true', help='Cache models in RAM to quickly switch between them'
    )

    args = parser.parse_args()

    launch_kwargs = {}
    launch_kwargs['server_name'] = args.listen

    if args.cache:
        MODELS = {}
    ui(
        username=args.username,
        password=args.password,
        inbrowser=args.inbrowser,
        server_port=args.server_port,
        share=args.share,
        listen=args.listen
    )
