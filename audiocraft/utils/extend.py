from tabnanny import verbose
import torch
import math
from audiocraft.models import MusicGen
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageColor
import string
import tempfile
import os
import textwrap
import requests
from io import BytesIO
from huggingface_hub import hf_hub_download
import librosa
import gradio as gr
import re
from tqdm import tqdm


INTERRUPTING = False

def separate_audio_segments(audio, segment_duration=30, overlap=1):
    sr, audio_data = audio[0], audio[1]
    
    segment_samples = sr * segment_duration
    total_samples = max(min((len(audio_data) // segment_samples), 25), 0)
    overlap_samples = sr * overlap

    segments = []
    start_sample = 0
    # handle the case where the audio is shorter than the segment duration
    if total_samples == 0:
        total_samples = 1
        segment_samples = len(audio_data)
        overlap_samples = 0
    while total_samples >= segment_samples:
        # Collect the segment
        # the end sample is the start sample plus the segment samples, 
        # the start sample, after 0, is minus the overlap samples to account for the overlap        
        end_sample = start_sample + segment_samples
        segment = audio_data[start_sample:end_sample]
        segments.append((sr, segment))

        start_sample += segment_samples - overlap_samples
        total_samples -= segment_samples

    # Collect the final segment
    if total_samples > 0:
        segment = audio_data[-segment_samples:]
        segments.append((sr, segment))
    print(f"separate_audio_segments: {len(segments)} segments of length {segment_samples // sr} seconds")
    return segments

def generate_music_segments(text, melody, seed, MODEL, duration:int=10, overlap:int=1, segment_duration:int=30, prompt_index:int=0, harmony_only:bool= False, excerpt_duration:float=3.5, progress= gr.Progress(track_tqdm=True)):
    # generate audio segments
    melody_segments = separate_audio_segments(melody, segment_duration, 0) 
    
    # Create lists to store the melody tensors for each segment
    melodys = []
    output_segments = []
    last_chunk = []
    text += ", seed=" + str(seed)
    prompt_segment = None
    # prevent hacking
    duration = min(duration, 720)
    overlap =  min(overlap, 15)
    
    # Calculate the total number of segments
    total_segments = max(math.ceil(duration / segment_duration),1)
    #calculate duration loss from segment overlap
    duration_loss = max(total_segments - 1,0) * math.ceil(overlap / 2)
    #calc excess duration
    excess_duration = segment_duration - (total_segments * segment_duration - duration)
    print(f"total Segments to Generate: {total_segments} for {duration} seconds. Each segment is {segment_duration} seconds. Excess {excess_duration} Overlap Loss {duration_loss}")
    duration += duration_loss
    pbar = tqdm(total=total_segments*2, desc="Generating segments", leave=False)
    while excess_duration + duration_loss > segment_duration:
        total_segments += 1
        #calculate duration loss from segment overlap
        duration_loss += math.ceil(overlap / 2)
        #calc excess duration
        excess_duration = segment_duration - (total_segments * segment_duration - duration)
        print(f"total Segments to Generate: {total_segments} for {duration} seconds. Each segment is {segment_duration} seconds. Excess {excess_duration} Overlap Loss {duration_loss}")
        if excess_duration + duration_loss > segment_duration:
            duration += duration_loss
            duration_loss = 0
    pbar.update(1)
    total_segments = min(total_segments, (720 // segment_duration))

    # If melody_segments is shorter than total_segments, repeat the segments until the total_segments is reached
    if len(melody_segments) < total_segments:
        #fix melody_segments
        for i in range(total_segments - len(melody_segments)):
            segment = melody_segments[i]
            melody_segments.append(segment)
            pbar.update(1)
        print(f"melody_segments: {len(melody_segments)} fixed")

    # Iterate over the segments to create list of Melody tensors
    for segment_idx in range(total_segments):
        if INTERRUPTING:
            return [], duration
        print(f"segment {segment_idx + 1} of {total_segments} \r")

        if harmony_only:
            # REMOVE PERCUSION FROM MELODY
            # Apply HPSS using librosa
            verse_harmonic, verse_percussive = librosa.effects.hpss(melody_segments[segment_idx][1])
            # Convert the separated components back to torch.Tensor
            #harmonic_tensor = torch.from_numpy(verse_harmonic)
            #percussive_tensor = torch.from_numpy(verse_percussive)
            sr, verse = melody_segments[segment_idx][0], torch.from_numpy(verse_harmonic).to(MODEL.device).float().t().unsqueeze(0)
        else:
            sr, verse = melody_segments[segment_idx][0], torch.from_numpy(melody_segments[segment_idx][1]).to(MODEL.device).float().t().unsqueeze(0)

        print(f"shape:{verse.shape} dim:{verse.dim()}")
        #if verse is 2D, add 3rd dimension
        if verse.dim() == 2:
           verse = verse[None]
        verse = verse[..., :int(sr * MODEL.lm.cfg.dataset.segment_duration)]

        # Reduce the length of verse to sr * excerpt_duration
        if ("style" in MODEL.name):
            verse = verse[:, :, :int(sr * excerpt_duration)]

        # Append the segment to the melodys list
        melodys.append(verse)
        pbar.update(1)
    pbar.close()  
    torch.manual_seed(seed)

    # If user selects a prompt segment, generate a new prompt segment to use on all segments
    #default to the first segment for prompt conditioning
    prompt_verse = melodys[0]
    if prompt_index > 0:
        # Get a prompt segment from the selected verse, normally the first verse
        prompt_verse = melodys[prompt_index if prompt_index <= (total_segments - 1) else (total_segments -1)]
       
    # set the prompt segment MODEL generation params
    MODEL.set_generation_params(
        use_sampling=True,
        top_k=MODEL.generation_params["top_k"],
        top_p=MODEL.generation_params["top_p"],
        temperature=MODEL.generation_params["temp"],
        cfg_coef=MODEL.generation_params["cfg_coef"],
        cfg_coef_beta=MODEL.generation_params["cfg_coef_beta"],
        duration=segment_duration,
        two_step_cfg=False,
        rep_penalty=0.5,
    )
    if ("style" in MODEL.name):
        MODEL.set_style_conditioner_params(
            eval_q=MODEL.lm.condition_provider.conditioners.self_wav.eval_q, # integer between 1 and 6
            excerpt_length=excerpt_duration, # the length in seconds that is taken by the model in the provided excerpt, can be between 1.5 and 4.5 seconds but it has to be shortest to the length of the provided conditioning
        )

    # Generate a new prompt segment. This will be applied to all segments for consistency
    print(f"Generating New Prompt Segment: {text} from verse {prompt_index}\r")
    prompt_segment = MODEL.generate_with_all(
        descriptions=[text],
        melody_wavs=prompt_verse,
        sample_rate=sr,
        progress=False,
        prompt=None,
    )       

    for idx, verse in tqdm(enumerate(melodys), total=len(melodys), desc="Generating melody segments"):
        if INTERRUPTING:
            return output_segments, duration

        print(f'Segment duration: {segment_duration}, duration: {duration}, overlap: {overlap} Overlap Loss: {duration_loss}')
        # Compensate for the length of final segment
        if ((idx + 1) == len(melodys)) or (duration < segment_duration):
            mod_duration = max(min(duration, segment_duration),1)
            print(f'Modify verse length, duration: {duration}, overlap: {overlap} Overlap Loss: {duration_loss} to mod duration: {mod_duration}')
            MODEL.set_generation_params(
                use_sampling=True,
                top_k=MODEL.generation_params["top_k"],
                top_p=MODEL.generation_params["top_p"],
                temperature=MODEL.generation_params["temp"],
                cfg_coef=MODEL.generation_params["cfg_coef"],
                cfg_coef_beta=MODEL.generation_params["cfg_coef_beta"],
                duration=mod_duration,
                two_step_cfg=False,
                rep_penalty=0.5,
            )

            if ("style" in MODEL.name):
                MODEL.set_style_conditioner_params(
                    eval_q=MODEL.lm.condition_provider.conditioners.self_wav.eval_q, # integer between 1 and 6
                    excerpt_length=min(excerpt_duration, mod_duration), # the length in seconds that is taken by the model in the provided excerpt, can be between 1.5 and 4.5 seconds but it has to be shortest to the length of the provided conditioning
                )

            try:
                # get last chunk
                verse = verse[:, :, -mod_duration*MODEL.sample_rate:]
                prompt_segment = prompt_segment[:, :, -mod_duration*MODEL.sample_rate:]
            except:
                # get first chunk
                verse = verse[:, :, :mod_duration*MODEL.sample_rate] 
                prompt_segment = prompt_segment[:, :, :mod_duration*MODEL.sample_rate]
      
            
        print(f"Generating New Melody Segment {idx + 1}: {text}\r")
        output, tokens = MODEL.generate_with_all(
            descriptions=[text],
            melody_wavs=verse,
            sample_rate=sr,
            progress=True,
            prompt=prompt_segment,
            return_tokens = True
        )
        # If user selects a prompt segment, use the prompt segment for all segments
        # Otherwise, use the previous segment as the prompt
        if prompt_index < 0:
            if harmony_only:
                # REMOVE PERCUSION FROM MELODY
                # Apply HPSS using librosa
                verse_harmonic, verse_percussive = librosa.effects.hpss(output.detach().cpu().numpy())
                # Convert the separated components back to torch.Tensor
                #harmonic_tensor = torch.from_numpy(verse_harmonic)
                #percussive_tensor = torch.from_numpy(verse_percussive)
                verse = torch.from_numpy(verse_harmonic).to(MODEL.device).float()
                # if verse is 2D, add extra dimension
                if verse.dim() == 2:
                   verse = verse[None]
                output = verse
            prompt_segment = output

        # Append the generated output to the list of segments
        #output_segments.append(output[:, :segment_duration])
        output_segments.append(output)
        print(f"output_segments: {len(output_segments)}: shape: {output.shape} dim {output.dim()}")
        #track duration
        if duration > segment_duration:
            duration -= segment_duration
    return output_segments, excess_duration

def save_image(image):
    """
    Saves a PIL image to a temporary file and returns the file path.

    Parameters:
    - image: PIL.Image
        The PIL image object to be saved.

    Returns:
    - str or None: The file path where the image was saved,
        or None if there was an error saving the image.

    """
    temp_dir = tempfile.gettempdir()
    temp_file = tempfile.NamedTemporaryFile(suffix=".png", dir=temp_dir, delete=False)
    temp_file.close()
    file_path = temp_file.name

    try:
        image.save(file_path)
        
    except Exception as e:
        print("Unable to save image:", str(e))
        return None
    finally:
        return file_path

def detect_color_format(color):
    """
    Detects if the color is in RGB, RGBA, or hex format,
    and converts it to an RGBA tuple with integer components.

    Args:
        color (str or tuple): The color to detect.

    Returns:
        tuple: The color in RGBA format as a tuple of 4 integers.

    Raises:
        ValueError: If the input color is not in a recognized format.
    """
    # Handle color as a tuple of floats or integers
    if isinstance(color, tuple):
        if len(color) == 3 or len(color) == 4:
            # Ensure all components are numbers
            if all(isinstance(c, (int, float)) for c in color):
                r, g, b = color[:3]
                a = color[3] if len(color) == 4 else 255
                return (
                    max(0, min(255, int(round(r)))),
                    max(0, min(255, int(round(g)))),
                    max(0, min(255, int(round(b)))),
                    max(0, min(255, int(round(a * 255)) if a <= 1 else round(a))),
                )
        else:
            raise ValueError(f"Invalid color tuple length: {len(color)}")
    # Handle hex color codes
    if isinstance(color, str):
        color = color.strip()
        # Try to use PIL's ImageColor
        try:
            rgba = ImageColor.getcolor(color, "RGBA")
            return rgba
        except ValueError:
            pass
        # Handle 'rgba(r, g, b, a)' string format
        rgba_match = re.match(r'rgba\(\s*([0-9.]+),\s*([0-9.]+),\s*([0-9.]+),\s*([0-9.]+)\s*\)', color)
        if rgba_match:
            r, g, b, a = map(float, rgba_match.groups())
            return (
                max(0, min(255, int(round(r)))),
                max(0, min(255, int(round(g)))),
                max(0, min(255, int(round(b)))),
                max(0, min(255, int(round(a * 255)) if a <= 1 else round(a))),
            )
        # Handle 'rgb(r, g, b)' string format
        rgb_match = re.match(r'rgb\(\s*([0-9.]+),\s*([0-9.]+),\s*([0-9.]+)\s*\)', color)
        if rgb_match:
            r, g, b = map(float, rgb_match.groups())
            return (
                max(0, min(255, int(round(r)))),
                max(0, min(255, int(round(g)))),
                max(0, min(255, int(round(b)))),
                255,
            )

    # If none of the above conversions work, raise an error
    raise ValueError(f"Invalid color format: {color}")

def hex_to_rgba(hex_color):
    try:
        if hex_color.startswith("#"):
            clean_hex = hex_color.replace('#','')
            # Use a generator expression to convert pairs of hexadecimal digits to integers and create a tuple
            rgba = tuple(int(clean_hex[i:i+2], 16) for i in range(0, len(clean_hex),2))
        else:
            rgba = tuple(map(int,detect_color_format(hex_color)))
    except ValueError:
        # If the hex color is invalid, default to yellow
        rgba = (255,255,0,255)
    return rgba

def load_font(font_name, font_size=16):
    """
    Load a font using the provided font name and font size.

    Parameters:
        font_name (str): The name of the font to load. Can be a font name recognized by the system, a URL to download the font file,
            a local file path, or a Hugging Face model hub identifier.
        font_size (int, optional): The size of the font. Default is 16.

    Returns:
        ImageFont.FreeTypeFont: The loaded font object.

    Notes:
        This function attempts to load the font using various methods until a suitable font is found. If the provided font_name
        cannot be loaded, it falls back to a default font.

        The font_name can be one of the following:
        - A font name recognized by the system, which can be loaded using ImageFont.truetype.
        - A URL pointing to the font file, which is downloaded using requests and then loaded using ImageFont.truetype.
        - A local file path to the font file, which is loaded using ImageFont.truetype.
        - A Hugging Face model hub identifier, which downloads the font file from the Hugging Face model hub using hf_hub_download
          and then loads it using ImageFont.truetype.

    Example:
        font = load_font("Arial.ttf", font_size=20)
    """
    font = None
    if not "http" in font_name:
        try:
            font = ImageFont.truetype(font_name, font_size)
        except (FileNotFoundError, OSError):
            print("Font not found. Using Hugging Face download..\n")

        if font is None:
            try:
                font_path = ImageFont.truetype(hf_hub_download(repo_id=os.environ.get('SPACE_ID', ''), filename="assets/" + font_name, repo_type="space"), encoding="UTF-8")        
                font = ImageFont.truetype(font_path, font_size)
            except (FileNotFoundError, OSError):
                print("Font not found. Trying to download from local assets folder...\n")
        if font is None:
            try:
                font = ImageFont.truetype("assets/" + font_name, font_size)
            except (FileNotFoundError, OSError):
                print("Font not found. Trying to download from URL...\n")

    if font is None:
        try:
            req = requests.get(font_name)
            font = ImageFont.truetype(BytesIO(req.content), font_size)       
        except (FileNotFoundError, OSError):
             print(f"Font not found: {font_name} Using default font\n")

    if font:
        print(f"Font loaded {font.getname()}")
    else:
        font = ImageFont.load_default()
    return font


def add_settings_to_image(title: str = "title", description: str = "", width: int = 768, height: int = 512, background_path: str = "", font: str = "arial.ttf", font_color: str = "#ffffff", font_size: int = 28, progress=gr.Progress(track_tqdm=True)):
    # Create a new RGBA image with the specified dimensions
    image = Image.new("RGBA", (width, height), (255, 255, 255, 0))
    # If a background image is specified, open it and paste it onto the image
    if background_path == "":
        background = Image.new("RGBA", (width, height), (255, 255, 255, 255))
    else:
        background = Image.open(background_path).convert("RGBA")

    #Convert font color to RGBA tuple
    font_color = hex_to_rgba(font_color)
    print(f"Font Color: {font_color}\n")

    # Calculate the center coordinates for placing the text
    text_x = width // 2
    text_y = height // 2
    # Draw the title text at the center top
    title_font = load_font(font, font_size)  # Replace with your desired font and size

    title_text = '\n'.join(textwrap.wrap(title, width // 12))
    title_x, title_y, title_text_width, title_text_height = title_font.getbbox(title_text)
    title_x = max(text_x - (title_text_width // 2), title_x, 0)
    title_y = text_y - (height // 2) + 10  # 10 pixels padding from the top
    title_draw = ImageDraw.Draw(image)
    title_draw.multiline_text((title_x, title_y), title, fill=font_color, font=title_font, align="center")
    # Draw the description text two lines below the title
    description_font = load_font(font, int(font_size * 2 / 3))  # Replace with your desired font and size
    description_text = '\n'.join(textwrap.wrap(description, width // 12))
    description_x, description_y, description_text_width, description_text_height = description_font.getbbox(description_text)
    description_x = max(text_x - (description_text_width // 2), description_x, 0)
    description_y = title_y + title_text_height + 20  # 20 pixels spacing between title and description
    description_draw = ImageDraw.Draw(image)
    description_draw.multiline_text((description_x, description_y), description_text, fill=font_color, font=description_font, align="center")
    # Calculate the offset to center the image on the background
    bg_w, bg_h = background.size
    offset = ((bg_w - width) // 2, (bg_h - height) // 2)
    # Paste the image onto the background
    background.paste(image, offset, mask=image)

    # Save the image and return the file path
    return save_image(background)