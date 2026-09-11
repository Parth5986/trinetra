from huggingface_hub import list_repo_files

repo_id = "alan-turing-institute/turing-synthetic-radar-dataset"

files = list_repo_files(
    repo_id=repo_id,
    repo_type="dataset"
)

for file in files:
    print(file)