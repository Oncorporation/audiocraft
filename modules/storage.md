# Storage Module (`modules/storage.py`) Usage Guide

The `storage.py` module provides helper functions for:
- Generating permalinks for 3D viewer projects.
- Uploading files in batches to a Hugging Face repository.
- Managing URL shortening by storing (short URL, full URL) pairs in a JSON file on the repository.
- Retrieving full URLs from short URL IDs and vice versa.
- Handle specific file types for 3D models, images, video and audio.

## Key Functions

### 1. `generate_permalink(valid_files, base_url_external, permalink_viewer_url="surn-3d-viewer.hf.space")`
- **Purpose:**  
  Given a list of file paths, it looks for exactly one model file (with an extension defined in `model_extensions`) and exactly two image files (extensions defined in `image_extensions`). If the criteria are met, it returns a permalink URL built from the base URL and query parameters.
- **Usage Example:**from modules.storage import generate_permalink

valid_files = [
    "models/3d_model.glb",
    "images/model_texture.png",
    "images/model_depth.png"
]
base_url_external = "https://huggingface.co/datasets/Surn/Storage/resolve/main/saved_models/my_model"
permalink = generate_permalink(valid_files, base_url_external)
if permalink:
    print("Permalink:", permalink)
### 2. `generate_permalink_from_urls(model_url, hm_url, img_url, permalink_viewer_url="surn-3d-viewer.hf.space")`
- **Purpose:**  
  Constructs a permalink URL by combining individual URLs for a 3D model (`model_url`), height map (`hm_url`), and image (`img_url`) into a single URL with corresponding query parameters.
- **Usage Example:**from modules.storage import generate_permalink_from_urls

model_url = "https://example.com/model.glb"
hm_url = "https://example.com/heightmap.png"
img_url = "https://example.com/source.png"

permalink = generate_permalink_from_urls(model_url, hm_url, img_url)
print("Generated Permalink:", permalink)
### 3. `upload_files_to_repo(files, repo_id, folder_name, create_permalink=False, repo_type="dataset", permalink_viewer_url="surn-3d-viewer.hf.space")`
- **Purpose:**  
  Uploads a batch of files (each file represented as a path string) to a specified Hugging Face repository (e.g. `"Surn/Storage"`) under a given folder.
  The function's return type is `Union[Dict[str, Any], List[Tuple[Any, str]]]`.
  - When `create_permalink` is `True` and exactly three valid files (one model and two images) are provided, the function returns a dictionary:```
{
    "response": <upload_folder_response>,
    "permalink": "<full_permalink_url>",
    "short_permalink": "<shortened_permalink_url_with_sid>"
}
```  - Otherwise (or if `create_permalink` is `False` or conditions for permalink creation are not met), it returns a list of tuples, where each tuple is `(upload_folder_response, individual_file_link)`.
  - If no valid files are provided, it returns an empty list `[]` (this case should ideally also return the dictionary with empty/None values for consistency, but currently returns `[]` as per the code).
- **Usage Example:**

  **a. Uploading with permalink creation:**from modules.storage import upload_files_to_repo

files_for_permalink = [
    "local/path/to/model.glb",
    "local/path/to/heightmap.png",
    "local/path/to/image.png"
]
repo_id = "Surn/Storage" # Make sure this is defined, e.g., from constants or environment variables
folder_name = "my_new_model_with_permalink"

upload_result = upload_files_to_repo(
    files_for_permalink, 
    repo_id, 
    folder_name, 
    create_permalink=True
)

if isinstance(upload_result, dict):
    print("Upload Response:", upload_result.get("response"))
    print("Full Permalink:", upload_result.get("permalink"))
    print("Short Permalink:", upload_result.get("short_permalink"))
elif upload_result: # Check if list is not empty
    print("Upload Response for individual files:")
    for res, link in upload_result:
        print(f"  Response: {res}, Link: {link}")
else:
    print("No files uploaded or error occurred.")
  **b. Uploading without permalink creation (or if conditions for permalink are not met):**from modules.storage import upload_files_to_repo

files_individual = [
    "local/path/to/another_model.obj",
    "local/path/to/texture.jpg"
]
repo_id = "Surn/Storage"
folder_name = "my_other_uploads"

upload_results_list = upload_files_to_repo(
    files_individual, 
    repo_id, 
    folder_name, 
    create_permalink=False # Or if create_permalink=True but not 1 model & 2 images
)

if upload_results_list: # Will be a list of tuples
    print("Upload results for individual files:")
    for res, link in upload_results_list:
        print(f"  Upload Response: {res}, File Link: {link}")
else:
    print("No files uploaded or error occurred.")
### 4. URL Shortening Functions: `gen_full_url(...)` and Helpers
The module also enables URL shortening by managing a JSON file (e.g. `shortener.json`) in a Hugging Face repository. It supports CRUD-like operations:
- **Read:** Look up the full URL using a provided short URL ID.
- **Create:** Generate a new short URL ID for a full URL if no existing mapping exists.
- **Update/Conflict Handling:**  
If both short URL ID and full URL are provided, it checks consistency and either confirms or reports a conflict.

#### `gen_full_url(short_url=None, full_url=None, repo_id=None, repo_type="dataset", permalink_viewer_url="surn-3d-viewer.hf.space", json_file="shortener.json")`
- **Purpose:**  
  Based on which parameter is provided, it retrieves or creates a mapping between a short URL ID and a full URL.  
  - If only `short_url` (the ID) is given, it returns the corresponding `full_url`.  
  - If only `full_url` is given, it looks up an existing `short_url` ID or generates and stores a new one.  
  - If both are given, it validates and returns the mapping or an error status.
- **Returns:** A tuple `(status_message, result_url)`, where `status_message` indicates the outcome (e.g., `"success_retrieved_full"`, `"created_short"`) and `result_url` is the relevant URL (full or short ID).
- **Usage Examples:**

  **a. Convert a full URL into a short URL ID:**from modules.storage import gen_full_url
from modules.constants import HF_REPO_ID, SHORTENER_JSON_FILE # Assuming these are defined

full_permalink = "https://surn-3d-viewer.hf.space/?3d=https%3A%2F%2Fexample.com%2Fmodel.glb&hm=https%3A%2F%2Fexample.com%2Fheightmap.png&image=https%3A%2F%2Fexample.com%2Fsource.png"

status, short_id = gen_full_url(
    full_url=full_permalink, 
    repo_id=HF_REPO_ID, 
    json_file=SHORTENER_JSON_FILE
)
print("Status:", status)
if status == "created_short" or status == "success_retrieved_short":
    print("Shortened URL ID:", short_id)
    # Construct the full short URL for sharing:
    # permalink_viewer_url = "surn-3d-viewer.hf.space" # Or from constants
    # shareable_short_url = f"https://{permalink_viewer_url}/?sid={short_id}"
    # print("Shareable Short URL:", shareable_short_url)
  **b. Retrieve the full URL from a short URL ID:**from modules.storage import gen_full_url
from modules.constants import HF_REPO_ID, SHORTENER_JSON_FILE # Assuming these are defined

short_id_to_lookup = "aBcDeFg1"  # Example short URL ID

status, retrieved_full_url = gen_full_url(
    short_url=short_id_to_lookup, 
    repo_id=HF_REPO_ID, 
    json_file=SHORTENER_JSON_FILE
)
print("Status:", status)
if status == "success_retrieved_full":
    print("Retrieved Full URL:", retrieved_full_url)
## Notes
- **Authentication:** All functions that interact with Hugging Face Hub use the HF API token defined as `HF_API_TOKEN` in `modules/constants.py`. Ensure this environment variable is correctly set.
- **Constants:** Functions like `gen_full_url` and `upload_files_to_repo` (when creating short links) rely on `HF_REPO_ID` and `SHORTENER_JSON_FILE` from `modules/constants.py` for the URL shortening feature.
- **File Types:** Only files with extensions included in `upload_file_types` (a combination of `model_extensions` and `image_extensions` from `modules/constants.py`) are processed by `upload_files_to_repo`.
- **Repository Configuration:** When using URL shortening and file uploads, ensure that the specified Hugging Face repository (e.g., defined by `HF_REPO_ID`) exists and that you have write permissions.
- **Temporary Directory:** `upload_files_to_repo` temporarily copies files to a local directory (configured by `TMPDIR` in `modules/constants.py`) before uploading.
- **Error Handling:** Functions include basic error handling (e.g., catching `RepositoryNotFoundError`, `EntryNotFoundError`, JSON decoding errors, or upload issues) and print messages to the console for debugging. Review function return values to handle these cases appropriately in your application.

---

This guide provides the essential usage examples for interacting with the storage and URL-shortening functionality. You can integrate these examples into your application or use them as a reference when extending functionality.
