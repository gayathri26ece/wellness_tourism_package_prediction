from huggingface_hub import HfApi
import os

# Retrieve the Hugging Face token from secrets.
HF_TOKEN = os.getenv("HF_TOKEN")

# Initialize API client with the retrieved token
api = HfApi(token=HF_TOKEN)

api.upload_folder(
    folder_path="tourism_project/deployment",     # the local folder containing your files
    # replace with your repoid
    repo_id = "gayathriL/wellness_tourism_package_prediction",          # the target repo

    repo_type="space",                      # dataset, model, or space
    path_in_repo="",                          # optional: subfolder path inside the repo
)
