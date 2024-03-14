"""
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
"""

__version__ = "0.2.0"

import json
import os
import shutil
import warnings
from datetime import datetime
from functools import cache
from pathlib import Path
from typing import Callable, Dict, List, Tuple
from uuid import uuid4

import gradio as gr
import numpy as np
import requests
from filelock import FileLock
from PIL.Image import Image
import filetype
import wave
from mutagen.mp3 import MP3, EasyMP3
import torchaudio
import subprocess


def setup(folder_path: str | Path | None = None) -> None:
    user_history = _UserHistory()
    user_history.folder_path = _resolve_folder_path(folder_path)
    user_history.initialized = True


def render() -> None:
    user_history = _UserHistory()

    # initialize with default config
    if not user_history.initialized:
        print("Initializing user history with default config. Use `user_history.setup(...)` to customize folder_path.")
        setup()

    # Render user history tab
    gr.Markdown(
        "## Your past generations\n\nLog in to keep a gallery of your previous generations. Your history will be saved"
        " and available on your next visit. Make sure to export your images from time to time as this gallery may be"
        " deleted in the future."
    )

    if os.getenv("SYSTEM") == "spaces" and not os.path.exists("/data"):
        gr.Markdown(
            "**⚠️ Persistent storage is disabled, meaning your history will be lost if the Space gets restarted."
            " Only the Space owner can setup a Persistent Storage. If you are not the Space owner, consider"
            " duplicating this Space to set your own storage.⚠️**"
        )

    with gr.Row():
        gr.LoginButton(min_width=250)
        #gr.LogoutButton(min_width=250)
        refresh_button = gr.Button(
            "Refresh",
            icon="./assets/icon_refresh.png",
        )
        export_button = gr.Button(
            "Export",
            icon="./assets/icon_download.png",
        )
        delete_button = gr.Button(
            "Delete history",
            icon="./assets/icon_delete.png",
        )

    # "Export zip" row (hidden by default)
    with gr.Row():
        export_file = gr.File(file_count="single", file_types=[".zip"], label="Exported history", visible=False)

    # "Config deletion" row (hidden by default)
    with gr.Row():
        confirm_button = gr.Button("Confirm delete all history", variant="stop", visible=False)
        cancel_button = gr.Button("Cancel", visible=False)

    # Gallery
    gallery = gr.Gallery(
        label="Past images",
        show_label=True,
        elem_id="gradio_user_history_gallery",
        object_fit="contain",
        columns=5,
        height=600,
        preview=False,
        show_share_button=False,
        show_download_button=False,
    )
    gr.Markdown(
        "User history is powered by"
        " [Wauplin/gradio-user-history](https://huggingface.co/spaces/Wauplin/gradio-user-history). Integrate it to"
        " your own Space in just a few lines of code!"
    )
    gallery.attach_load_event(_fetch_user_history, every=None)

    # Interactions
    refresh_button.click(fn=_fetch_user_history, inputs=[], outputs=[gallery], queue=False)
    export_button.click(fn=_export_user_history, inputs=[], outputs=[export_file], queue=False)

    # Taken from https://github.com/gradio-app/gradio/issues/3324#issuecomment-1446382045
    delete_button.click(
        lambda: [gr.update(visible=True), gr.update(visible=True)],
        outputs=[confirm_button, cancel_button],
        queue=False,
    )
    cancel_button.click(
        lambda: [gr.update(visible=False), gr.update(visible=False)],
        outputs=[confirm_button, cancel_button],
        queue=False,
    )
    confirm_button.click(_delete_user_history).then(
        lambda: [gr.update(visible=False), gr.update(visible=False)],
        outputs=[confirm_button, cancel_button],
        queue=False,
    )

    # Admin section (only shown locally or when logged in as Space owner)
    _admin_section()


def save_image(
    profile: gr.OAuthProfile | None,
    image: Image | np.ndarray | str | Path,
    label: str | None = None,
    metadata: Dict | None = None,
):
    # Ignore images from logged out users
    if profile is None:
        return
    username = profile["preferred_username"]

    # Ignore images if user history not used
    user_history = _UserHistory()
    if not user_history.initialized:
        warnings.warn(
            "User history is not set in Gradio demo. Saving image is ignored. You must use `user_history.render(...)`"
            " first."
        )
        return

    # Copy image to storage
    image_path = _copy_image(image, dst_folder=user_history._user_images_path(username))

    # Save new image + metadata
    if metadata is None:
        metadata = {}
    if "datetime" not in metadata:
        metadata["datetime"] = str(datetime.now())
    data = {"path": str(image_path), "label": label, "metadata": metadata}
    with user_history._user_lock(username):
        with user_history._user_jsonl_path(username).open("a") as f:
            f.write(json.dumps(data) + "\n")
            
def save_file(
    profile: gr.OAuthProfile | None,
    image: Image | np.ndarray | str | Path | None = None,
    video: str | Path | None = None,
    audio: str | Path | None = None,
    document: str | Path | None = None,
    label: str | None = None,
    metadata: Dict | None = None,
):
    # Ignore files from logged out users
    if profile is None:
        return
    username = profile["preferred_username"]

    # Ignore files if user history not used
    user_history = _UserHistory()
    if not user_history.initialized:
        warnings.warn(
            "User history is not set in Gradio demo. Saving files is ignored. You must use `user_history.render(...)`"
            " first."
        )
        return
    
    # Save new files + metadata
    if metadata is None:
        metadata = {}
    if "datetime" not in metadata:
        metadata["datetime"] = str(datetime.now())
        
    # Copy image to storage
    image_path = None
    if image is not None:
        image_path = _copy_image(image, dst_folder=user_history._user_images_path(username))
        image_path = _add_metadata(image_path, metadata)

    # Copy video to storage
    if video is not None:    
        video_path = _copy_file(video, dst_folder=user_history._user_file_path(username, "videos"))
        video_path = _add_metadata(video_path, metadata)

    # Copy audio to storage
    if audio is not None:     
        audio_path = _copy_file(audio, dst_folder=user_history._user_file_path(username, "audios"))
        audio_path = _add_metadata(audio_path, metadata)
    
    # Copy document to storage
    if document is not None:     
        document_path = _copy_file(document, dst_folder=user_history._user_file_path(username, "documents"))
        document_path = _add_metadata(document_path, metadata)

    # Save Json file
    data = {"image_path": str(image_path), "video_path": str(video_path), "audio_path": str(audio_path), "document_path": str(document_path), "label": label, "metadata": metadata}
    with user_history._user_lock(username):
        with user_history._user_jsonl_path(username).open("a") as f:
            f.write(json.dumps(data) + "\n")


#############
# Internals #
#############


class _UserHistory(object):
    _instance = None
    initialized: bool = False
    folder_path: Path

    def __new__(cls):
        # Using singleton pattern => we don't want to expose an object (more complex to use) but still want to keep
        # state between `render` and `save_image` calls.
        if cls._instance is None:
            cls._instance = super(_UserHistory, cls).__new__(cls)
        return cls._instance

    def _user_path(self, username: str) -> Path:
        path = self.folder_path / username
        path.mkdir(parents=True, exist_ok=True)
        return path

    def _user_lock(self, username: str) -> FileLock:
        """Ensure history is not corrupted if concurrent calls."""
        return FileLock(self.folder_path / f"{username}.lock")  # lock outside of folder => better when exporting ZIP

    def _user_jsonl_path(self, username: str) -> Path:
        return self._user_path(username) / "history.jsonl"

    def _user_images_path(self, username: str) -> Path:
        path = self._user_path(username) / "images"
        path.mkdir(parents=True, exist_ok=True)
        return path
    
    def _user_file_path(self, username: str, filetype: str = "images") -> Path:        
        path = self._user_path(username) / filetype
        path.mkdir(parents=True, exist_ok=True)
        return path
   
    

def _fetch_user_history(profile: gr.OAuthProfile | None) -> List[Tuple[str, str]]:
    """Return saved history for that user, if it exists."""
    # Cannot load history for logged out users
    if profile is None:
        return []
    username = profile["preferred_username"]

    user_history = _UserHistory()
    if not user_history.initialized:
        warnings.warn("User history is not set in Gradio demo. You must use `user_history.render(...)` first.")
        return []

    with user_history._user_lock(username):
        # No file => no history saved yet
        jsonl_path = user_history._user_jsonl_path(username)
        if not jsonl_path.is_file():
            return []

        # Read history
        images = []
        for line in jsonl_path.read_text().splitlines():
            data = json.loads(line)
            images.append((data["path"], data["label"] or ""))
        return list(reversed(images))


def _export_user_history(profile: gr.OAuthProfile | None) -> Dict | None:
    """Zip all history for that user, if it exists and return it as a downloadable file."""
    # Cannot load history for logged out users
    if profile is None:
        return None
    username = profile["preferred_username"]

    user_history = _UserHistory()
    if not user_history.initialized:
        warnings.warn("User history is not set in Gradio demo. You must use `user_history.render(...)` first.")
        return None

    # Zip history
    with user_history._user_lock(username):
        path = shutil.make_archive(
            str(_archives_path() / f"history_{username}"), "zip", user_history._user_path(username)
        )

    return gr.update(visible=True, value=path)


def _delete_user_history(profile: gr.OAuthProfile | None) -> None:
    """Delete all history for that user."""
    # Cannot load history for logged out users
    if profile is None:
        return
    username = profile["preferred_username"]

    user_history = _UserHistory()
    if not user_history.initialized:
        warnings.warn("User history is not set in Gradio demo. You must use `user_history.render(...)` first.")
        return

    with user_history._user_lock(username):
        shutil.rmtree(user_history._user_path(username))


####################
# Internal helpers #
####################


def _copy_image(image: Image | np.ndarray | str | Path, dst_folder: Path) -> Path:
    try:
        """Copy image to the images folder."""
        # Already a path => copy it
        if isinstance(image, str):
            image = Path(image)
        if isinstance(image, Path):
            dst = dst_folder / f"{uuid4().hex}_{Path(image).name}"  # keep file ext
            shutil.copyfile(image, dst)
            return dst

        # Still a Python object => serialize it
        if isinstance(image, np.ndarray):
            image = Image.fromarray(image)
        if isinstance(image, Image):
            dst = dst_folder / f"Path(file).name}_{uuid4().hex}.png"
            image.save(dst)
            return dst

        raise ValueError(f"Unsupported image type: {type(image)}")
    
    except Exception as e:
        print(f"An error occurred: {e}")
        if not isinstance(dst, Path):
            dst = Path(image)
        return dst  # Return the original file_location if an error occurs

def _copy_file(file: any | np.ndarray | str | Path, dst_folder: Path) -> Path:
    try:
        """Copy file to the appropriate folder."""
        # Already a path => copy it
        if isinstance(file, str):
            file = Path(file)
        if isinstance(file, Path):
            dst = dst_folder / f"{file.stem}_{uuid4().hex}{file.suffix}"  # keep file ext
            shutil.copyfile(file, dst)
            return dst

        # Still a Python object => serialize it
        if isinstance(file, np.ndarray):
            file = Image.fromarray(file)
            dst = dst_folder / f"{file.filename}_{uuid4().hex}{file.suffix}"
            file.save(dst)
            return dst

        # try other file types
        kind = filetype.guess(file)
        if kind is not None:
            dst = dst_folder / f"{Path(file).stem}_{uuid4().hex}.{kind.extension}"
            shutil.copyfile(file, dst)
            return dst
        raise ValueError(f"Unsupported file type: {type(file)}")

    except Exception as e:
        print(f"An error occurred: {e}")
        if not isinstance(dst, Path):
            dst = Path(file)
        return dst  # Return the original file_location if an error occurs


def _add_metadata(file_location: Path, metadata: Dict[str, Any]) -> Path:
    try:
        file_type = file_location.suffix
        valid_file_types = [".wav", ".mp3", ".mp4", ".png"]
        if file_type not in valid_file_types:
            raise ValueError("Invalid file type. Valid file types are .wav, .mp3, .mp4, .png")

        if file_type == ".wav":
            # Open and process .wav file
            with wave.open(file_location, 'rb') as wav_file:
                # Get the current metadata
                current_metadata = {key: value for key, value in wav_file.getparams()._asdict().items() if isinstance(value, (int, float))}
                
                # Update metadata
                current_metadata.update(metadata)

                # Reopen the WAV file in write mode
                with wave.open(file_location, 'wb') as wav_output_file:
                    # Set the new metadata
                    wav_output_file.setparams(wav_file.getparams())

            # Save the WAV file (overwriting the previous version)
            wav_output_file.close()
        elif file_type == ".mp3":
            # Open and process .mp3 file
            audio = EasyMP3(file_location)

            # Add metadata to the file
            for key, value in metadata.items():
                audio[key] = value

            # Save the MP3 file (overwriting the previous version)
            audio.save()
        elif file_type == ".mp4":
            # Open and process .mp4 file
            # Add metadata to the file
            wav_file_location = file_location.with_suffix(".wav")
            wave_exists = wav_file_location.exists()
            if not wave_exists:
                # Use torchaudio to create the WAV file if it doesn't exist
                audio, sample_rate = torchaudio.load(file_location, normalize=True)
                torchaudio.save(wav_file_location, audio, sample_rate, format='wav')

            # Use ffmpeg to add metadata to the video file
            metadata_args = [f"{key}={value}" for key, value in metadata.items()]
            ffmpeg_metadata = ":".join(metadata_args)
            ffmpeg_cmd = f'ffmpeg -i "{file_location}" -i "{wav_file_location}" -map 0:v:0 -map 1:a:0 -c:v copy -c:a aac -metadata "{ffmpeg_metadata}" "{file_location}"'
            subprocess.run(ffmpeg_cmd, shell=True, check=True)

            # Remove temporary WAV file
            if not wave_exists:
                wav_file_location.unlink()
        elif file_type == ".png":
            # Open and process .png file
            image = Image.open(file_location)
            exif_data = image.info.get("exif", {})
            exif_data.update(metadata)
            # Add metadata to the file
            image.save(file_location, exif=exif_data)

        return file_location  # Return the path to the modified file

    except Exception as e:
        print(f"An error occurred: {e}")
        return file_location  # Return the original file_location if an error occurs
   
def _resolve_folder_path(folder_path: str | Path | None) -> Path:
    if folder_path is not None:
        return Path(folder_path).expanduser().resolve()

    if os.getenv("SYSTEM") == "spaces" and os.path.exists("/data"):  # Persistent storage is enabled!
        return Path("/data") / "_user_history"

    # Not in a Space or Persistent storage not enabled => local folder
    return Path("_user_history").resolve()


def _archives_path() -> Path:
    # Doesn't have to be on persistent storage as it's only used for download
    path = Path(__file__).parent / "_user_history_exports"
    path.mkdir(parents=True, exist_ok=True)
    return path


#################
# Admin section #
#################


def _admin_section() -> None:
    title = gr.Markdown()
    title.attach_load_event(_display_if_admin(), every=None)


def _display_if_admin() -> Callable:
    def _inner(profile: gr.OAuthProfile | None) -> str:
        if profile is None:
            return ""
        if profile["preferred_username"] in _fetch_admins():
            return _admin_content()
        return ""

    return _inner


def _admin_content() -> str:
    return f"""
## Admin section

Running on **{os.getenv("SYSTEM", "local")}** (id: {os.getenv("SPACE_ID")}). {_get_msg_is_persistent_storage_enabled()}

Admins: {', '.join(_fetch_admins())}

{_get_nb_users()} user(s), {_get_nb_images()} image(s)

### Configuration

History folder: *{_UserHistory().folder_path}*

Exports folder: *{_archives_path()}*

### Disk usage

{_disk_space_warning_message()}
"""


def _get_nb_users() -> int:
    user_history = _UserHistory()
    if not user_history.initialized:
        return 0
    if user_history.folder_path is not None and user_history.folder_path.exists():
        return len([path for path in user_history.folder_path.iterdir() if path.is_dir()])
    return 0


def _get_nb_images() -> int:
    user_history = _UserHistory()
    if not user_history.initialized:
        return 0
    if user_history.folder_path is not None and user_history.folder_path.exists():
        return len([path for path in user_history.folder_path.glob("*/images/*")])
    return 0


def _get_msg_is_persistent_storage_enabled() -> str:
    if os.getenv("SYSTEM") == "spaces":
        if os.path.exists("/data"):
            return "Persistent storage is enabled."
        else:
            return (
                "Persistent storage is not enabled. This means that user histories will be deleted when the Space is"
                " restarted. Consider adding a Persistent Storage in your Space settings."
            )
    return ""


def _disk_space_warning_message() -> str:
    user_history = _UserHistory()
    if not user_history.initialized:
        return ""

    message = ""
    if user_history.folder_path is not None:
        total, used, _ = _get_disk_usage(user_history.folder_path)
        message += f"History folder: **{used / 1e9 :.0f}/{total / 1e9 :.0f}GB** used ({100*used/total :.0f}%)."

    total, used, _ = _get_disk_usage(_archives_path())
    message += f"\n\nExports folder: **{used / 1e9 :.0f}/{total / 1e9 :.0f}GB** used ({100*used/total :.0f}%)."

    return f"{message.strip()}"


def _get_disk_usage(path: Path) -> Tuple[int, int, int]:
    for path in [path] + list(path.parents):  # first check target_dir, then each parents one by one
        try:
            return shutil.disk_usage(path)
        except OSError:  # if doesn't exist or can't read => fail silently and try parent one
            pass
    return 0, 0, 0


@cache
def _fetch_admins() -> List[str]:
    # Running locally => fake user is admin
    if os.getenv("SYSTEM") != "spaces":
        return ["FakeGradioUser"]

    # Running in Space but no space_id => ???
    space_id = os.getenv("SPACE_ID")
    if space_id is None:
        return ["Unknown"]

    # Running in Space => try to fetch organization members
    # Otherwise, it's not an organization => namespace is the user
    namespace = space_id.split("/")[0]
    response = requests.get(f"https://huggingface.co/api/organizations/{namespace}/members")
    if response.status_code == 200:
        return sorted((member["user"] for member in response.json()), key=lambda x: x.lower())
    return [namespace]
