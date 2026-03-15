import os
import uuid 

def get_id():
    run_id = str(os.environ.get("RUN_ID", "")).strip()
    if run_id:
        return run_id
    return str(uuid.uuid4())[:10]

__all__ = ['get_id']
