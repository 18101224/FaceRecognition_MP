from datetime import datetime 
import uuid
def get_exp_id(args):
    exp_id = args.server+str(uuid.uuid4())
    return exp_id