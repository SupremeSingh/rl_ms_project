"""Download the frozen encoder at an immutable revision; never create pilot datasets."""
import argparse
import json
from pathlib import Path
from huggingface_hub import HfApi, snapshot_download
from slice_critic.config import MODEL_ID, immutable_revision
from slice_critic.storage import model_identity

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--revision',help='Pinned model commit SHA; first setup resolves latest only when omitted')
    args=p.parse_args();path=Path('configs/assets.json')
    saved=json.loads(path.read_text()) if path.exists() else None
    revision=args.revision or (saved['model_revision'] if saved else HfApi().model_info(MODEL_ID).sha)
    if not immutable_revision(revision): raise ValueError('Require an immutable model commit SHA')
    if saved and (saved['model_id']!=MODEL_ID or saved['model_revision']!=revision):
        raise ValueError('Existing model pin differs; preserve it in another checkout before replacing')
    snapshot_download(MODEL_ID,revision=revision,local_dir='models/qwen-math')
    files=model_identity('models/qwen-math')
    path.write_text(json.dumps(dict(model_id=MODEL_ID,model_revision=revision,model_hashes=files),indent=2)+'\n')
