from datetime import datetime 

def get_exp_id(args):
    now = datetime.now()
    exp_id = args.server+now.strftime('%m%d%H%M%S%f')[:12]  # mmddhhmmssmm
    return exp_id