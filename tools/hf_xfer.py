# Temporary transfer L40 -> intern02 via a public HF dataset repo holding only an AES-256 encrypted file (owner OK 2026-10-16).
# Usage: hf_xfer.py up FILE  -> prints repo id ; hf_xfer.py rm REPO_ID
import sys, secrets
from huggingface_hub import create_repo, upload_file, delete_repo
if sys.argv[1] == "up":
    rid = f"NothingSpecialSiri/xfer-{secrets.token_hex(4)}"
    create_repo(rid, repo_type="dataset", private=False)
    upload_file(path_or_fileobj=sys.argv[2], path_in_repo="data.bin", repo_id=rid, repo_type="dataset", commit_message="temporary encrypted transfer")
    print("REPO", rid)
else:
    delete_repo(sys.argv[2], repo_type="dataset"); print("deleted", sys.argv[2])
