---
title: UnlimitedMusicGen
emoji: 🎼
colorFrom: gray
colorTo: red
sdk: gradio
sdk_version: 5.23.3
python_version: 3.12.8
app_file: app.py
pinned: true
license: creativeml-openrail-m
tags:
- musicgen
- unlimited
- user history
- metadata
hf_oauth: true
disable_embedding: true
short_description: 'unlimited Audio generation with a few added features '
thumbnail: >-
  https://cdn-uploads.huggingface.co/production/uploads/6346595c9e5f0fe83fc60444/Z8E8OaKV84zuVAvvGpMDJ.png
---

[arxiv]: https://arxiv.org/abs/2306.05284
[musicgen_samples]: https://ai.honu.io/papers/musicgen/
Check out the configuration reference at https://huggingface.co/docs/hub/spaces-config-reference

# UnlimitedMusicGen
Charles Fettinger's modification of the Audiocraft project to enable unlimited Audio generation. I have added a few features to the original project to enable this. I have also added a few features to the gradio interface to make it easier to use.

Please review my other AI relalated spaces at https://huggingface.co/Surn

Check your video's generative metadata with https://mediaarea.net/en/MediaInfo

Also note that I wrote an extension to Gradio for the waveform in the video after v4.48.0 removed it.

The key update here is in the extend utility. We segment melody input and then condition the next segment with current tensors and tensors from the current time in the conditioning melody file.
This allows us to make entire songs that follow the same arraigement of the original.

**Thank you Huggingface for the community grant to run this project**!!

# Audiocraft
![docs badge](https://github.com/facebookresearch/audiocraft/workflows/audiocraft_docs/badge.svg)
![linter badge](https://github.com/facebookresearch/audiocraft/workflows/audiocraft_linter/badge.svg)
![tests badge](https://github.com/facebookresearch/audiocraft/workflows/audiocraft_tests/badge.svg)

Audiocraft is a PyTorch library for deep learning research on audio generation. At the moment, it contains the code for MusicGen, a state-of-the-art controllable text-to-music model.

## MusicGen

Audiocraft provides the code and models for MusicGen, [a simple and controllable model for music generation][arxiv]. MusicGen is a single stage auto-regressive
Transformer model trained over a 32kHz <a href="https://github.com/facebookresearch/encodec">EnCodec tokenizer</a> with 4 codebooks sampled at 50 Hz. Unlike existing methods like [MusicLM](https://arxiv.org/abs/2301.11325), MusicGen doesn't require a self-supervised semantic representation, and it generates
all 4 codebooks in one pass. By introducing a small delay between the codebooks, we show we can predict
them in parallel, thus having only 50 auto-regressive steps per second of audio.
Check out our [sample page][musicgen_samples] or test the available demo!

<a target="_blank" href="https://colab.research.google.com/drive/1-Xe9NCdIs2sCUbiSmwHXozK6AAhMm7_i?usp=sharing">
  <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/>
</a>
<a target="_blank" href="https://huggingface.co/spaces/facebook/MusicGen">
  <img src="https://huggingface.co/datasets/huggingface/badges/raw/main/open-in-hf-spaces-sm.svg" alt="Open in HugginFace"/>
</a>
<br>

We use 20K hours of licensed music to train MusicGen. Specifically, we rely on an internal dataset of 10K high-quality music tracks, and on the ShutterStock and Pond5 music data.

## Installation
Audiocraft requires Python 3.9, PyTorch 2.1.0, and a GPU with at least 16 GB of memory (for the medium-sized model). To install Audiocraft, you can run the following:

```shell
# Best to make sure you have torch installed first, in particular before installing xformers.
# Don't run this if you already have PyTorch installed.
pip install 'torch>=2.1'
# Then proceed to one of the following
pip install -U audiocraft  # stable release
pip install -U git+https://git@github.com/facebookresearch/audiocraft#egg=audiocraft  # bleeding edge
pip install -e .  # or if you cloned the repo locally
```

## Usage
We offer a number of way to interact with MusicGen:
1. A demo is also available on the [`facebook/MusicGen`  HuggingFace Space](https://huggingface.co/spaces/Surn/UnlimitedMusicGen) (huge thanks to all the HF team for their support).
2. You can run the Gradio demo in Colab: [colab notebook](https://colab.research.google.com/drive/1-Xe9NCdIs2sCUbiSmwHXozK6AAhMm7_i?usp=sharing).
3. You can use the gradio demo locally by running `python app.py`.
4. You can play with MusicGen by running the jupyter notebook at [`demo.ipynb`](./demo.ipynb) locally (if you have a GPU).
5. Checkout [@camenduru Colab page](https://github.com/camenduru/MusicGen-colab) which is regularly
  updated with contributions from @camenduru and the community.
6. Finally, MusicGen is available in 🤗 Transformers from v4.31.0 onwards, see section [🤗 Transformers Usage](#-transformers-usage) below.

### More info about Top-k, Top-p, Temperature and Classifier Free Guidance from ChatGPT
6. Finally, MusicGen is available in 🤗 Transformers from v4.31.0 onwards, see section [🤗 Transformers Usage](#-transformers-usage) below.


Top-k: Top-k is a parameter used in text generation models, including music generation models. It determines the number of most likely next tokens to consider at each step of the generation process. The model ranks all possible tokens based on their predicted probabilities, and then selects the top-k tokens from the ranked list. The model then samples from this reduced set of tokens to determine the next token in the generated sequence. A smaller value of k results in a more focused and deterministic output, while a larger value of k allows for more diversity in the generated music.

Top-p (or nucleus sampling): Top-p, also known as nucleus sampling or probabilistic sampling, is another method used for token selection during text generation. Instead of specifying a fixed number like top-k, top-p considers the cumulative probability distribution of the ranked tokens. It selects the smallest possible set of tokens whose cumulative probability exceeds a certain threshold (usually denoted as p). The model then samples from this set to choose the next token. This approach ensures that the generated output maintains a balance between diversity and coherence, as it allows for a varying number of tokens to be considered based on their probabilities.

Temperature: Temperature is a parameter that controls the randomness of the generated output. It is applied during the sampling process, where a higher temperature value results in more random and diverse outputs, while a lower temperature value leads to more deterministic and focused outputs. In the context of music generation, a higher temperature can introduce more variability and creativity into the generated music, but it may also lead to less coherent or structured compositions. On the other hand, a lower temperature can produce more repetitive and predictable music.

Classifier-Free Guidance: Classifier-Free Guidance refers to a technique used in some music generation models where a separate classifier network is trained to provide guidance or control over the generated music. This classifier is trained on labeled data to recognize specific musical characteristics or styles. During the generation process, the output of the generator model is evaluated by the classifier, and the generator is encouraged to produce music that aligns with the desired characteristics or style. This approach allows for more fine-grained control over the generated music, enabling users to specify certain attributes they want the model to capture.

These parameters, such as top-k, top-p, temperature, and classifier-free guidance, provide different ways to influence the output of a music generation model and strike a balance between creativity, diversity, coherence, and control. The specific values for these parameters can be tuned based on the desired outcome and user preferences.

## API

We provide a simple API and 10 pre-trained models. The pre trained models are:
- `small`: 300M model, text to music only - [🤗 Hub](https://huggingface.co/facebook/musicgen-small)
- `medium`: 1.5B model, text to music only - [🤗 Hub](https://huggingface.co/facebook/musicgen-medium)
- `melody`: 1.5B model, text to music and text+melody to music - [🤗 Hub](https://huggingface.co/facebook/musicgen-melody)
- `large`: 3.3B model, text to music only - [🤗 Hub](https://huggingface.co/facebook/musicgen-large)
- `melody large` (3.3B), text to music, and text+melody to music # see: [🤗 Hub](https://huggingface.co/facebook/musicgen-melody-large)
- `small stereo` (300M), text to music, # see: [🤗 Hub](https://huggingface.co/facebook/musicgen-small)
- `medium stereo` (1.5B), text to music, # see: [🤗 Hub](https://huggingface.co/facebook/musicgen-stereo-medium)
- `melody stereo` (1.5B) text to music and text+melody to music, # see: [🤗 Hub](https://huggingface.co/facebook/musicgen-stereo-melody)
- `large stereo` (3.3B), text to music, # see: [🤗 Hub](https://huggingface.co/facebook/musicgen-stereo-large)
- `melody large stereo` (3.3B), text to music, and text+melody to music # see: [🤗 Hub](https://huggingface.co/facebook/musicgen-stereo-melody-large)

We observe the best trade-off between quality and compute with the `medium` or `melody` model.
In order to use MusicGen locally **you must have a GPU**. We recommend 16GB of memory, but smaller
GPUs will be able to generate short sequences, or longer sequences with the `small` model.

**Note**: Please make sure to have [ffmpeg](https://ffmpeg.org/download.html) installed when using newer version of `torchaudio`.
You can install it with:
```
apt-get install ffmpeg
```

See after a quick example for using the API.

```python
import torchaudio
from audiocraft.models import MusicGen
from audiocraft.data.audio import audio_write

model = MusicGen.get_pretrained('melody')
model.set_generation_params(duration=8)  # generate 8 seconds.
wav = model.generate_unconditional(4)    # generates 4 unconditional audio samples
descriptions = ['happy rock', 'energetic EDM', 'sad jazz']
wav = model.generate(descriptions)  # generates 3 samples.

melody, sr = torchaudio.load('./assets/bach.mp3')
# generates using the melody from the given audio and the provided descriptions.
wav = model.generate_with_chroma(descriptions, melody[None].expand(3, -1, -1), sr)

for idx, one_wav in enumerate(wav):
    # Will save under {idx}.wav, with loudness normalization at -14 db LUFS.
    audio_write(f'{idx}', one_wav.cpu(), model.sample_rate, strategy="loudness", loudness_compressor=True)
```
## 🤗 Transformers Usage

MusicGen is available in the 🤗 Transformers library from version 4.31.0 onwards, requiring minimal dependencies 
and additional packages. Steps to get started:

1. First install the 🤗 [Transformers library](https://github.com/huggingface/transformers) from main:

```
pip install git+https://github.com/huggingface/transformers.git
```

2. Run the following Python code to generate text-conditional audio samples:

```py
from transformers import AutoProcessor, MusicgenForConditionalGeneration


processor = AutoProcessor.from_pretrained("facebook/musicgen-small")
model = MusicgenForConditionalGeneration.from_pretrained("facebook/musicgen-small")

inputs = processor(
    text=["80s pop track with bassy drums and synth", "90s rock song with loud guitars and heavy drums"],
    padding=True,
    return_tensors="pt",
)

audio_values = model.generate(**inputs, max_new_tokens=256)
```

3. Listen to the audio samples either in an ipynb notebook:

```py
from IPython.display import Audio

sampling_rate = model.config.audio_encoder.sampling_rate
Audio(audio_values[0].numpy(), rate=sampling_rate)
```

Or save them as a `.wav` file using a third-party library, e.g. `scipy`:

```py
import scipy

sampling_rate = model.config.audio_encoder.sampling_rate
scipy.io.wavfile.write("musicgen_out.wav", rate=sampling_rate, data=audio_values[0, 0].numpy())
```

For more details on using the MusicGen model for inference using the 🤗 Transformers library, refer to the 
[MusicGen docs](https://huggingface.co/docs/transformers/main/en/model_doc/musicgen) or the hands-on 
[Google Colab](https://colab.research.google.com/github/sanchit-gandhi/notebooks/blob/main/MusicGen.ipynb).

## User History

User History is a plugin that you can add to your Spaces to cache generated images for your users.

Key features:
- 🤗 Sign in with Hugging Face
- Save generated image, video, audio and document files with their metadata: prompts, timestamp, hyper-parameters, etc.
- Export your history as zip.
- Delete your history to respect privacy.
- Compatible with Persistent Storage for long-term storage.
- Admin panel to check configuration and disk usage .

Useful links:
- Demo: https://huggingface.co/spaces/Wauplin/gradio-user-history
- README: https://huggingface.co/spaces/Wauplin/gradio-user-history/blob/main/README.md
- Source file: https://huggingface.co/spaces/Wauplin/gradio-user-history/blob/main/user_history.py
- Discussions: https://huggingface.co/spaces/Wauplin/gradio-user-history/discussions

![Image preview](./assets/screenshot.png)

## Model Card

See [the model card page](./MODEL_CARD.md).

## FAQ

#### Will the training code be released?

Yes. We will soon release the training code for MusicGen and EnCodec.


#### I need help on Windows

@FurkanGozukara made a complete tutorial for [Audiocraft/MusicGen on Windows](https://youtu.be/v-YpvPkhdO4)

#### I need help for running the demo on Colab

Check [@camenduru tutorial on Youtube](https://www.youtube.com/watch?v=EGfxuTy9Eeo).

## Citation
```
@article{copet2023simple,
      title={Simple and Controllable Music Generation},
      author={Jade Copet and Felix Kreuk and Itai Gat and Tal Remez and David Kant and Gabriel Synnaeve and Yossi Adi and Alexandre Défossez},
      year={2023},
      journal={arXiv preprint arXiv:2306.05284},
}
```

## License
* The code in this repository is released under the MIT license as found in the [LICENSE file](LICENSE).
* The weights in this repository are released under the CC-BY-NC 4.0 license as found in the [LICENSE_weights file](LICENSE_weights).
[arxiv]: https://arxiv.org/abs/2306.05284

[arxiv]: https://arxiv.org/abs/2306.05284
[musicgen_samples]: https://ai.honu.io/papers/musicgen/
Check out the configuration reference at https://huggingface.co/docs/hub/spaces-config-reference