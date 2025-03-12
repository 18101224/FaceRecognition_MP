import wandb
from utils import get_exp_name
def init_wandb(args):
    with open(args.wandb_token,'r') as f:
        token = f.readline()[:-1]
    wandb.login(key=token)
    name = get_exp_name()
    wandb.init(
        project=f'KP_RPE pretrain {args.dataset_name}',
        name = name if not args.name else args.name,
        config={
            'learning_rate': args.learning_rate,
            'batch_size' : args.batch_size,
            'epochs' : args.n_epochs,
            'world_size' : args.world_size,
            'save_path' : f'checkpoint/{name}'
        }
    )