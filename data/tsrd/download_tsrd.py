from huggingface_hub import hf_hub_download, get_token, whoami
from huggingface_hub.errors import GatedRepoError
import os

repo_id = "alan-turing-institute/turing-synthetic-radar-dataset"

# Change this range to download fewer files, e.g. range(3, 31) = test_3..test_30
files_to_download = [f"archive/test/test_{i}.h5" for i in range(3, 151)]

# Download into data/tsrd/ so the HF paths (archive/test/test_N.h5) land
# exactly in data/tsrd/archive/test/ next to test_0/1/2.h5.
local_dir = os.path.dirname(os.path.abspath(__file__))
os.makedirs(local_dir, exist_ok=True)

# --- Resolve a Hugging Face read token -------------------------------
# Priority: HF_TOKEN env var  ->  saved login  ->  interactive prompt.
token = os.environ.get("HF_TOKEN") or get_token()
if not token:
    token = input("Paste your Hugging Face read token (hf_...): ").strip()
if not token:
    raise SystemExit("No token provided - cannot access the gated dataset.")

# Verify the token is valid before burning time on downloads.
try:
    user = whoami(token=token).get("name", "?")
    print(f"Authenticated as: {user}\n")
except Exception as e:
    raise SystemExit(
        f"\nToken rejected by Hugging Face: {e}\n"
        "Check your token at https://huggingface.co/settings/tokens (role = Read)."
    ) from e

# --- Download ---------------------------------------------------------
for filename in files_to_download:
    print(f"Downloading {filename}...")
    try:
        path = hf_hub_download(
            repo_id=repo_id,
            repo_type="dataset",
            filename=filename,
            local_dir=local_dir,
            token=token,
        )
    except GatedRepoError as e:
        msg = str(e)
        if "403" in msg:
            raise SystemExit(
                f"\nYour token is VALID but this account is NOT approved for the gated dataset.\n"
                f"Open https://huggingface.co/datasets/{repo_id}\n"
                "click 'Request access', wait for approval, then re-run this script.\n"
            ) from e
        raise SystemExit(
            f"\nToken was rejected (401) - double-check it in "
            "https://huggingface.co/settings/tokens\n"
        ) from e
    print(f"  OK -> {path}")

print(f"\nDone: {len(files_to_download)} files downloaded into {local_dir}")