from datetime import datetime 
import uuid

__all__ = ['get_exp_id']

def get_exp_id(args):
    exp_id = args.server+str(uuid.uuid4())
    return exp_id