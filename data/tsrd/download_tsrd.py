# from huggingface_hub import hf_hub_download
# import os
#
# repo_id = "alan-turing-institute/turing-synthetic-radar-dataset"
#
# files_to_download = [
#     "archive/test/test_0.h5",
#     "archive/test/test_1.h5",
#     "archive/test/test_2.h5",
# ]
#
# local_dir = ""
#
# os.makedirs(local_dir, exist_ok=True)
#
# for filename in files_to_download:
#     print(f"Downloading {filename}...")
#
#     path = hf_hub_download(
#         repo_id=repo_id,
#         repo_type="dataset",
#         filename=filename,
#         local_dir=local_dir,
#     )
#
#     print(f"Downloaded: {path}")

from huggingface_hub import hf_hub_download

repo_id = "alan-turing-institute/turing-synthetic-radar-dataset"

path = hf_hub_download(
    repo_id=repo_id,
    repo_type="dataset",
    filename="archive/test/test_0.h5",
    local_dir="data/tsrd",
)

print("\nFILE DOWNLOADED TO:")
print(path)