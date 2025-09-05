from analysis import Analysis, Compare, Analyze_backbone
from argparse import ArgumentParser

def get_args():
    args = ArgumentParser()
    args.add_argument('--dataset_path', type=str, required=True, help='Path to the dataset')
    args.add_argument('--dataset_name', type=str, required=True, help='Name of the dataset')
    args.add_argument('--model_type', type=str, required=False, help='Type of the model', choices=['resnet32', 'resnet50', 'resnext50'])
    args.add_argument('--aligner_path', type=str, default=None, help='Path to the aligner model or config')
    args.add_argument('--save_path', type=str, required=True, help='Directory to save results/plots')
    args.add_argument('--model_paths', type=str, nargs='+', required=False, default=None, help='Path(s) to the model checkpoint(s) (optional)')
    args.add_argument('--model_names', type=str, nargs='+', required=False, default=None, help='Name(s) of the model(s) (optional)')
    args.add_argument('--imb_factor', type=float, required=False, default=1, help='Image factor')
    args.add_argument('--mode',choices=['analysis','compare','backbone','dataset'],default='analysis')
    args.add_argument('--ckpt_type', type=str, required=False, choices=['best_acc','latest','best_macro_acc','best_acc_balanced' ])
    args.add_argument('--model_type', type=str, required=False, choices=['resnet32', 'resnet50', 'resnext50','ir50','kp_rpe'])
    return args.parse_args()

if __name__ == '__main__':
    args = get_args()
    if args.mode == 'compare':
        compare = Compare(args)
        compare.main()
    elif args.mode == 'backbone':
        backbone = Analyze_backbone(args)
        backbone.main()
    elif args.mode == 'analysis':
        analysis = Analysis(args)
        analysis.main()
    else:
        analysis = Analysis(args)
        analysis.main()
