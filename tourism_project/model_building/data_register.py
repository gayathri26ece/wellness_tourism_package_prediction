
from huggingface_hub import HfApi, create_repo
from huggingface_hub.utils import RepositoryNotFoundError, HfHubHTTPError
import os

# Retrieve the Hugging Face token from secrets.
token = os.getenv("HF_TOKEN")

# Initialize API client with the retrieved token
api = HfApi(token=token)

repo_id = "gayathriL/wellness_tourism_package_prediction"
repo_type = "dataset"

# Step 1: Check if the space exists
try:
    api.repo_info(repo_id=repo_id, repo_type=repo_type)
    print(f"Space '{repo_id}' already exists. Using it.")
except RepositoryNotFoundError:
    print(f"Space '{repo_id}' not found. Creating new space...")
    # Attempt to create the repository, passing the token explicitly
    create_repo(repo_id=repo_id, repo_type=repo_type, private=False, token=token)
    print(f"Space '{repo_id}' created.")

# Upload data, passing the token explicitly
api.upload_folder(
    folder_path="tourism_project/data",
    repo_id=repo_id,
    repo_type=repo_type,
    token=token
)
print(f"Data from 'tourism_project/data' uploaded to '{repo_id}'.")
