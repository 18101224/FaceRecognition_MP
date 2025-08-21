import math
import numpy as np
from torch.optim import lr_scheduler

def linear_rampup(current, rampup_length=0):
    if rampup_length == 0:
        return 1.0
    else:
        current = np.clip(current / rampup_length, 0.0, 1.0)
        return float(current)
    
def cos_decay(epoch, begin, end):
    return 0.5 * (1. + math.cos(math.pi * (epoch - begin + 1) / (end - begin)))

def adjust_learning_rate(optimizer, epoch, scheduler, args):
    if scheduler == None:
        if args.n_epochs == 200:
            if args.scheduler == 'cosine':
                epoch = epoch + 1
                if epoch <= args.warmup_epochs:
                    lr = args.learning_rate * epoch / args.warmup_epochs
                else:
                    lr = args.learning_rate * cos_decay(epoch=epoch, begin=args.warmup_epochs, end=args.n_epochs)
                
                for param_group in optimizer.param_groups:
                    param_group["lr"] = lr
                    
                return lr
        
            else: # warm-up
                epoch = epoch + 1
                if epoch <= args.warmup_epochs:
                    lr = args.learning_rate * epoch / args.warmup_epochs
                elif epoch > 180:
                    lr = args.learning_rate * args.gamma ** 2
                elif epoch > 160:
                    lr = args.learning_rate * args.gamma
                else:
                    lr = args.learning_rate

                for param_group in optimizer.param_groups:
                    param_group['lr'] = lr
                return lr
                
        elif args.n_epochs == 400:
            if 'BCL' in args.loss:
                epoch = epoch + 1
                if epoch <= args.warmup_epochs:
                    lr = args.learning_rate * epoch / args.warmup_epochs
                elif epoch > 380:
                    lr = args.learning_rate * args.gamma ** 2
                elif epoch > 360:
                    lr = args.learning_rate * args.gamma
                else:
                    lr = args.learning_rate

                for param_group in optimizer.param_groups:
                    param_group['lr'] = lr
                return lr
            else:
                epoch = epoch + 1
                if epoch <= args.warmup_epochs:
                    lr = args.learning_rate * epoch / args.warmup_epochs
                elif epoch > 360:
                    lr = args.learning_rate * args.gamma ** 2
                elif epoch > 320:
                    lr = args.learning_rate * args.gamma
                else:
                    lr = args.learning_rate

                for param_group in optimizer.param_groups:
                    param_group['lr'] = lr
                return lr
        else:
            return args.learning_rate
    else:
        scheduler.step()
        return optimizer.param_groups[0]['lr']
    
def get_scheduler(args, optimizer):
    if args.scheduler == 'cosine':
        return lr_scheduler.CosineAnnealingLR(optimizer, args.n_epochs, eta_min = args.learning_rate/100)
    elif args.scheduler == 'warmup':
        return None