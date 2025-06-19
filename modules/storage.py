# modules/storage.py
__version__ = "0.1.1" # Added version
import os
import urllib.parse
import tempfile
import shutil
import json
import base64
from huggingface_hub import login, upload_folder, hf_hub_download, HfApi
from huggingface_hub.utils import RepositoryNotFoundError, EntryNotFoundError
from modules.constants import HF_API_TOKEN, upload_file_types, model_extensions, image_extensions, audio_extensions, video_extensions, HF_REPO_ID, SHORTENER_JSON_FILE
from typing import Any, Dict, List, Tuple, Union

# see storage.md for detailed information about the storage module and its functions.

def generate_permalink(valid_files, base_url_external, permalink_viewer_url="surn-3d-viewer.hf.space"):
    """
    Given a list of valid files, checks if they contain exactly 1 model file and 2 image files.
    Constructs and returns a permalink URL with query parameters if the criteria is met.
    Otherwise, returns None.
    """
    model_link = None
    images_links = []
    audio_links = []
    video_links = []
    for f in valid_files:
        filename = os.path.basename(f)
        ext = os.path.splitext(filename)[1].lower()
        if ext in model_extensions:
            if model_link is None:
                model_link = f"{base_url_external}/{filename}"
        elif ext in image_extensions:
            images_links.append(f"{base_url_external}/{filename}")
        elif ext in audio_extensions:
            audio_links.append(f"{base_url_external}/{filename}")
        elif ext in video_extensions:
            video_links.append(f"{base_url_external}/{filename}")
    if model_link and len(images_links) == 2:
        # Construct a permalink to the viewer project with query parameters.
        permalink_viewer_url = f"https://{permalink_viewer_url}/"
        params = {"3d": model_link, "hm": images_links[0], "image": images_links[1]}
        query_str = urllib.parse.urlencode(params)
        return f"{permalink_viewer_url}?{query_str}"
    return None

def generate_permalink_from_urls(model_url, hm_url, img_url, permalink_viewer_url="surn-3d-viewer.hf.space"):
    """
    Constructs and returns a permalink URL with query string parameters for the viewer.
    Each parameter is passed separately so that the image positions remain consistent.
    
    Parameters:
        model_url (str): Processed URL for the 3D model.
        hm_url (str): Processed URL for the height map image.
        img_url (str): Processed URL for the main image.
        permalink_viewer_url (str): The base viewer URL.
    
    Returns:
        str: The generated permalink URL.
    """
    import urllib.parse
    params = {"3d": model_url, "hm": hm_url, "image": img_url}
    query_str = urllib.parse.urlencode(params)
    return f"https://{permalink_viewer_url}/?{query_str}"

def upload_files_to_repo(
    files: List[Any],
    repo_id: str,
    folder_name: str,
    create_permalink: bool = False,
    repo_type: str = "dataset",
    permalink_viewer_url: str = "surn-3d-viewer.hf.space"
) -> Union[Dict[str, Any], List[Tuple[Any, str]]]:
    """
    Uploads multiple files to a Hugging Face repository using a batch upload approach via upload_folder.

    Parameters:
        files (list): A list of file paths (str) to upload.
        repo_id (str): The repository ID on Hugging Face for storage, e.g. "Surn/Storage".
        folder_name (str): The subfolder within the repository where files will be saved.
        create_permalink (bool): If True and if exactly three files are uploaded (1 model and 2 images),
                                 returns a single permalink to the project with query parameters.
                                 Otherwise, returns individual permalinks for each file.
        repo_type (str): Repository type ("space", "dataset", etc.). Default is "dataset".
        permalink_viewer_url (str): The base viewer URL.

    Returns:
        Union[Dict[str, Any], List[Tuple[Any, str]]]:
            If create_permalink is True and files match the criteria:
                dict: {
                    "response": <upload response>,
                    "permalink": <full_permalink URL>,
                    "short_permalink": <shortened permalink URL>
                }
            Otherwise:
                list: A list of tuples (response, permalink) for each file.
    """
    # Log in using the HF API token.
    login(token=HF_API_TOKEN) # Corrected from HF_TOKEN to HF_API_TOKEN
    
    valid_files = []
    permalink_short = None
    
    # Ensure folder_name does not have a trailing slash.
    folder_name = folder_name.rstrip("/")
    
    # Filter for valid files based on allowed extensions.
    for f in files:
        file_name = f if isinstance(f, str) else f.name if hasattr(f, "name") else None
        if file_name is None:
            continue
        ext = os.path.splitext(file_name)[1].lower()
        if ext in upload_file_types:
            valid_files.append(f)
    
    if not valid_files:
        # Return a dictionary with None values for permalinks if create_permalink was True
        if create_permalink:
            return {
                "response": "No valid files to upload.",
                "permalink": None,
                "short_permalink": None
            }
        return [] 
    
    # Create a temporary directory; copy valid files directly into it.
    with tempfile.TemporaryDirectory(dir=os.getenv("TMPDIR", "/tmp")) as temp_dir:
        for file_path in valid_files:
            filename = os.path.basename(file_path)
            dest_path = os.path.join(temp_dir, filename)
            shutil.copy(file_path, dest_path)
        
        # Batch upload all files in the temporary folder.
        # Files will be uploaded under the folder (path_in_repo) given by folder_name.
        response = upload_folder(
            folder_path=temp_dir,
            repo_id=repo_id,
            repo_type=repo_type,
            path_in_repo=folder_name,
            commit_message="Batch upload files"
        )
    
    # Construct external URLs for each uploaded file.
    base_url_external = f"https://huggingface.co/datasets/{repo_id}/resolve/main/{folder_name}"
    individual_links = []
    for file_path in valid_files:
        filename = os.path.basename(file_path)
        link = f"{base_url_external}/{filename}"
        individual_links.append(link)
    
    # If permalink creation is requested and exactly 3 valid files are provided,
    # try to generate a permalink using generate_permalink().
    if create_permalink: # No need to check len(valid_files) == 3 here, generate_permalink will handle it
        permalink = generate_permalink(valid_files, base_url_external, permalink_viewer_url)
        if permalink:
            status, short_id = gen_full_url(
                full_url=permalink,
                repo_id=HF_REPO_ID, # This comes from constants
                json_file=SHORTENER_JSON_FILE # This comes from constants
            )
            if status in ["created_short", "success_retrieved_short", "exists_match"]:
                permalink_short = f"https://{permalink_viewer_url}/?sid={short_id}"
            else: # Shortening failed or conflict not resolved to a usable short_id
                permalink_short = None 
                print(f"URL shortening status: {status} for {permalink}")

            return {
                "response": response,
                "permalink": permalink,
                "short_permalink": permalink_short
            }
        else: # generate_permalink returned None (criteria not met)
            return {
                "response": response, # Still return upload response
                "permalink": None,
                "short_permalink": None
            }

    # Otherwise, return individual tuples for each file.
    return [(response, link) for link in individual_links]

def _generate_short_id(length=8):
    """Generates a random base64 URL-safe string."""
    return base64.urlsafe_b64encode(os.urandom(length * 2))[:length].decode('utf-8')

def _get_json_from_repo(repo_id, json_file_name, repo_type="dataset"):
    """Downloads and loads the JSON file from the repo. Returns empty list if not found or error."""
    try:
        login(token=HF_API_TOKEN)
        json_path = hf_hub_download(
            repo_id=repo_id,
            filename=json_file_name,
            repo_type=repo_type,
            token=HF_API_TOKEN  # Added token for consistency, though login might suffice
        )
        with open(json_path, 'r') as f:
            data = json.load(f)
        os.remove(json_path) # Clean up downloaded file
        return data
    except RepositoryNotFoundError:
        print(f"Repository {repo_id} not found.")
        return []
    except EntryNotFoundError:
        print(f"JSON file {json_file_name} not found in {repo_id}. Initializing with empty list.")
        return []
    except json.JSONDecodeError:
        print(f"Error decoding JSON from {json_file_name}. Returning empty list.")
        return []
    except Exception as e:
        print(f"An unexpected error occurred while fetching {json_file_name}: {e}")
        return []

def _upload_json_to_repo(data, repo_id, json_file_name, repo_type="dataset"):
    """Uploads the JSON data to the specified file in the repo."""
    try:
        login(token=HF_API_TOKEN)
        api = HfApi()
        # Use a temporary directory specified by TMPDIR or default to system temp
        temp_dir_for_json = os.getenv("TMPDIR", tempfile.gettempdir())
        os.makedirs(temp_dir_for_json, exist_ok=True)

        with tempfile.NamedTemporaryFile(mode="w+", delete=False, suffix=".json", dir=temp_dir_for_json) as tmp_file:
            json.dump(data, tmp_file, indent=2)
            tmp_file_path = tmp_file.name
        
        api.upload_file(
            path_or_fileobj=tmp_file_path,
            path_in_repo=json_file_name,
            repo_id=repo_id,
            repo_type=repo_type,
            commit_message=f"Update {json_file_name}"
        )
        os.remove(tmp_file_path) # Clean up temporary file
        return True
    except Exception as e:
        print(f"Failed to upload {json_file_name} to {repo_id}: {e}")
        if 'tmp_file_path' in locals() and os.path.exists(tmp_file_path):
            os.remove(tmp_file_path) # Ensure cleanup on error too
        return False

def _find_url_in_json(data, short_url=None, full_url=None):
    """
    Searches the JSON data.
    If short_url is provided, returns the corresponding full_url or None.
    If full_url is provided, returns the corresponding short_url or None.
    """
    if not data: # Handles cases where data might be None or empty
        return None
    if short_url:
        for item in data:
            if item.get("short_url") == short_url:
                return item.get("full_url")
    if full_url:
        for item in data:
            if item.get("full_url") == full_url:
                return item.get("short_url")
    return None

def _add_url_to_json(data, short_url, full_url):
    """Adds a new short_url/full_url pair to the data. Returns updated data."""
    if data is None: 
        data = []
    data.append({"short_url": short_url, "full_url": full_url})
    return data

def gen_full_url(short_url=None, full_url=None, repo_id=None, repo_type="dataset", permalink_viewer_url="surn-3d-viewer.hf.space", json_file="shortener.json"):
    """
    Manages short URLs and their corresponding full URLs in a JSON file stored in a Hugging Face repository.

    - If short_url is provided, attempts to retrieve and return the full_url.
    - If full_url is provided, attempts to retrieve an existing short_url or creates a new one, stores it, and returns the short_url.
    - If both are provided, checks for consistency or creates a new entry.
    - If neither is provided, or repo_id is missing, returns an error status.

    Returns:
        tuple: (status_message, result_url)
               status_message can be "success", "created", "exists", "error", "not_found".
               result_url is the relevant URL (short or full) or None if an error occurs or not found.
    """
    if not repo_id:
        return "error_repo_id_missing", None
    if not short_url and not full_url:
        return "error_no_input", None

    login(token=HF_API_TOKEN) # Ensure login at the beginning
    url_data = _get_json_from_repo(repo_id, json_file, repo_type)

    # Case 1: Only short_url provided (lookup full_url)
    if short_url and not full_url:
        found_full_url = _find_url_in_json(url_data, short_url=short_url)
        return ("success_retrieved_full", found_full_url) if found_full_url else ("not_found_short", None)

    # Case 2: Only full_url provided (lookup or create short_url)
    if full_url and not short_url:
        existing_short_url = _find_url_in_json(url_data, full_url=full_url)
        if existing_short_url:
            return "success_retrieved_short", existing_short_url
        else:
            # Create new short_url
            new_short_id = _generate_short_id()
            url_data = _add_url_to_json(url_data, new_short_id, full_url)
            if _upload_json_to_repo(url_data, repo_id, json_file, repo_type):
                return "created_short", new_short_id 
            else:
                return "error_upload", None

    # Case 3: Both short_url and full_url provided
    if short_url and full_url:
        found_full_for_short = _find_url_in_json(url_data, short_url=short_url)
        found_short_for_full = _find_url_in_json(url_data, full_url=full_url)

        if found_full_for_short == full_url: 
            return "exists_match", short_url 
        if found_full_for_short is not None and found_full_for_short != full_url: 
            return "error_conflict_short_exists_different_full", short_url
        if found_short_for_full is not None and found_short_for_full != short_url:
            return "error_conflict_full_exists_different_short", found_short_for_full
        
        # If short_url is provided and not found, or full_url is provided and not found,
        # or neither is found, then create a new entry with the provided short_url and full_url.
        # This effectively allows specifying a custom short_url if it's not already taken.
        url_data = _add_url_to_json(url_data, short_url, full_url)
        if _upload_json_to_repo(url_data, repo_id, json_file, repo_type):
            return "created_specific_pair", short_url
        else:
            return "error_upload", None
                
    return "error_unhandled_case", None # Should not be reached
