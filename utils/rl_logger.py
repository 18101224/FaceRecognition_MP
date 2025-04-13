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

def init_wandb_clothing1m(args):
    with open(args.wandb_token, 'r') as f:
        token = f.readline()[:-1]
    wandb.login(key=token)
    wandb.init(
        project='RL',
        name=args.name,
        config={'learning_rate' : args.learning_rate,
                'batch_size' : args.batch_size,
                'epochs' : args.n_epochs,
                'world_size' : args.world_size,
                'constant_cos_margin' : args.constant_margin,
                'adaptive_cos_margin' : args.adaptive_margin}
    )