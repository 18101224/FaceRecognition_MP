import wandb

def init_wandb(args):
    with open(args.wandb_token, 'r') as f:
        token = f.readline()[:-1]
    wandb.login(key=token)
    wandb.init(
        project='RL',
        name = args.name ,
        config = {'learning_rate' : args.learning_rate,
                  'batch_size' : args.batch_size,
                  'epochs' : args.n_epochs,
                  'opt' : args.opt,
                  'world_size' : args.world_size,
                  'init_classifier' : args.env_init_path,
                  'dataset' : args.dataset,
                  'ada_loss' : args.ada_loss,
                  'gamma' : args.gamma,
                  'gp' : args.gp ,
                  'k' : args.k,
                  'save_path' : args.save_path
                  }

    )
