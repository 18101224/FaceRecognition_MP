from datetime import datetime 
import random 
import time 

def get_exp_id(args):
    now = datetime.now()
    random_num = random.uniform(0, 10)
    time.sleep(random_num)
    print(f'random_num: {random_num}')
    exp_id = args.server+now.strftime('%m%d%H%M%S%f')[:12]  # mmddhhmmssmm
    return exp_id