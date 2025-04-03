# modules/version_info.py

from audiocraft import __version__ as audiocraft_version
import subprocess
import os
import sys
import gc
import gradio as gr

git = os.environ.get('GIT', "git")

def commit_hash():
    try:
        return subprocess.check_output([git, "rev-parse", "HEAD"], shell=False, encoding='utf8').strip()
    except Exception:
        return "<none>"

def get_xformers_version():
    try:
        import xformers
        return xformers.__version__
    except Exception:
        return "<none>"
def get_transformers_version():
    try:
        import transformers
        return transformers.__version__
    except Exception:
        return "<none>"

def get_accelerate_version():
    try:
        import accelerate
        return accelerate.__version__
    except Exception:
        return "<none>"
def get_safetensors_version():
    try:
        import safetensors
        return safetensors.__version__
    except Exception:
        return "<none>"
def get_diffusers_version():
    try:
        import diffusers
        return diffusers.__version__
    except Exception:
        return "<none>"

def get_torch_info():
    from torch import __version__ as torch_version_, version, cuda, backends
    device_type = initialize_cuda()
    if device_type == "cuda":
        try:
            info = [torch_version_, f"CUDA Version:{version.cuda}", f"Available:{cuda.is_available()}", f"flash attention enabled: {backends.cuda.flash_sdp_enabled()}", f"Capabilities: {cuda.get_device_capability(0)}", f"Device Name: {cuda.get_device_name(0)}", f"Device Count: {cuda.device_count()}"]
            del torch_version_, version, cuda, backends
            return info
        except Exception:
            del torch_version_, version, cuda, backends
            return "<none>"
    else:
        return "Not Recognized"

def release_torch_resources():
    from torch import cuda
    # Clear the CUDA cache
    cuda.empty_cache()
    cuda.ipc_collect()
    # Delete any objects that are using GPU memory
    #for obj in gc.get_objects():
    #    if is_tensor(obj) or (hasattr(obj, 'data') and is_tensor(obj.data)):
    #        del obj
    # Run garbage collection
    del cuda
    gc.collect()
    

def initialize_cuda():
    from torch import cuda, version
    if cuda.is_available():
        device = cuda.device("cuda")
        print(f"CUDA is available. Using device: {cuda.get_device_name(0)} with CUDA version: {version.cuda}")
        result = "cuda"
    else:
        print("CUDA is not available. Using CPU.")
        result = "cpu"
    return result

def versions_html():
    from torch import __version__ as torch_version_
    python_version = ".".join([str(x) for x in sys.version_info[0:3]])
    commit = commit_hash()

    # Define the Toggle Dark Mode link with JavaScript
    toggle_dark_link = '''
        <a href="#" onclick="document.body.classList.toggle('dark'); return false;" style="cursor: pointer; text-decoration: underline;">
            Toggle Dark Mode
        </a>
    '''

    v_html = f"""
        version: <a href="https://huggingface.co/spaces/Surn/UnlimitedMusicGen/commit/{"huggingface" if commit == "<none>" else commit}" target="_blank">{"huggingface" if commit == "<none>" else commit}</a>
        &#x2000;•&#x2000;
        autocraft: {audiocraft_version}
        &#x2000;•&#x2000;
        python: <span title="{sys.version}">{python_version}</span>
        &#x2000;•&#x2000;
        torch: {torch_version_}
        &#x2000;•&#x2000;
        xformers: {get_xformers_version()}
        &#x2000;•&#x2000;
        transformers: {get_transformers_version()}
        &#x2000;•&#x2000;
        safetensors: {get_safetensors_version()}
        &#x2000;•&#x2000;
        gradio: {gr.__version__}
        &#x2000;•&#x2000;
        {toggle_dark_link}
        <br>
        Full GPU Info:{get_torch_info()}
        """
    del torch_version_
    return v_html