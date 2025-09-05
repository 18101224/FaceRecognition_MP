from .computation import *
from .plot import *
from .load import *
import torch, os
from copy import deepcopy
import sys; sys.path.append('..')
from models import  CosClassifier
from models.modules import resnet32_backbone as resnet32
from argparse import Namespace

feature_names = ['backbone_feat', 'cls_feat', 'bcl_feat', 'center_feat']

class Analysis:
    def __init__(self,args):
        self.args = args
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        #load dataset 
        self.datasets = load_dataset(self.args,dataset_path=self.args.dataset_path, dataset_name=self.args.dataset_name, imb_factor=self.args.imb_factor)
        self.loaders = load_loaders(self.datasets)
        self.save_path = self.args.save_path
        if not os.path.exists(self.save_path):
            os.makedirs(self.save_path,exist_ok=True)
        if self.args.mode == 'dataset':
            return 
        #load model
        model_ckpt = os.path.join(self.args.model_paths[0], f'{self.args.ckpt_type}.pth')
        log_ckpt = torch.load(os.path.join(self.args.model_paths[0], 'latest.pth'), weights_only=False)
        log_args = vars(log_ckpt['args']) if isinstance(log_ckpt['args'], Namespace) else log_ckpt['args']
        cur_args = vars(self.args) if isinstance(self.args, Namespace) else self.args
        self.args = Namespace(**{**log_args, **cur_args})
        self.model = get_model(self.args)
        self.model.load_state_dict(torch.load(model_ckpt, map_location=self.device,weights_only=False)['model_state_dict'])
        self.model = self.model.to(self.device)
        self.model.eval()
        self.aligner = load_aligner(self.args.aligner_path) if self.args.aligner_path else None
        # self.backbone_analysis = Analyze_backbone(self.args)
        self.backbone = self.model.backbone 
        self.weight = self.model.weight.T.detach().cpu().numpy()

    def analyze_dataset(self):
        plot_label_distribution(self.datasets[0].labels, self.datasets[1].labels, self.save_path, model_name=getattr(self.args, 'model_name', None))

    def analyze_model_performance(self):
        train_preds, train_labels, train_confs = get_predictions(self.model, self.loaders[0], self.aligner)
        valid_preds, valid_labels, valid_confs = get_predictions(self.model, self.loaders[1], self.aligner)
        valid_preds_balanced, valid_labels_balanced, valid_confs_balanced = get_predictions(self.model, self.loaders[2], self.aligner) if len(self.loaders) == 3 else (None, None, None)
        #compute confusion matrix
        valid_cm = compute_confusion_matrix(valid_preds, valid_labels)
        valid_normed_cm = normalize_confusion_matrix(valid_cm, valid_labels)

        valid_cm_balanced = compute_confusion_matrix(valid_preds_balanced, valid_labels_balanced) if valid_preds_balanced is not None else None
        valid_normed_cm_balanced = normalize_confusion_matrix(valid_cm_balanced, valid_labels_balanced) if valid_cm_balanced is not None else None

        #plot confusion matrix
        plot_confusion_matrix(valid_cm, valid_normed_cm, self.save_path, model_name=getattr(self.args, 'model_name', None))
        plot_confusion_matrix(valid_cm_balanced, valid_normed_cm_balanced, self.save_path, model_name=getattr(self.args, 'model_name', None),save_name='valid_confusion_matrix_balanced.png') if valid_cm_balanced is not None else None

        #compute angle matrix
        angle_matrix = compute_angle_matrix(self.model)
        plot_angle_with_confusion_matrix(angle=angle_matrix, conf=valid_normed_cm, save_path=self.save_path)
        plot_angle_with_confusion_matrix(angle=angle_matrix, conf=valid_normed_cm_balanced, save_path=self.save_path, save_name='valid_confusion_matrix_balanced.png') if valid_cm_balanced is not None else None
        error_rates = compute_error_rate_per_class(preds =valid_preds, labels = valid_labels)
        error_rates_balanced = compute_error_rate_per_class(preds =valid_preds_balanced, labels = valid_labels_balanced) if valid_preds_balanced is not None else None
        plot_class_num_and_accuracy(train_labels, error_rates, self.save_path, model_name=getattr(self.args, 'model_name', None))
        plot_class_num_and_accuracy(train_labels, error_rates_balanced, self.save_path, model_name=getattr(self.args, 'model_name', None), save_name='valid_confusion_matrix_balanced.png') if valid_cm_balanced is not None else None

        # Calculate validation accuracy
        correct_predictions = (valid_preds == valid_labels).sum()
        total_predictions = len(valid_labels)
        validation_accuracy = correct_predictions / total_predictions
        print(f"Validation Accuracy: {validation_accuracy * 100:.2f}%")

    def analyze_features(self):
        backbone = self.model.backbone
        train_features, train_labels = get_features(self.model, self.train_loader, self.aligner)
        valid_features, valid_labels = get_features(self.model, self.valid_loader, self.aligner)


        x = torch.randn((1,3,32,32)).to(self.device)
        _,_,centers_logits = self.model(x, True)
        centers_logits = centers_logits.detach().cpu().numpy()
        if False:
            train_features_tsne = get_tsne_features(train_features, output_dim=2)
            valid_features_tsne = get_tsne_features(valid_features, output_dim=2)
            train_centers = plot_tsne_features(train_features_tsne, train_labels, self.train_set.labels, self.save_path, original_features=train_features, model_name=getattr(self.args, 'model_name', None))
            valid_centers = plot_tsne_features(valid_features_tsne, valid_labels, self.valid_set.labels, self.save_path, original_features=valid_features, model_name=getattr(self.args, 'model_name', None))
        if not isinstance(train_features, list) :
            train_features = [train_features]
            valid_features = [valid_features]
        else:
            train_features = process_features(train_features)
            valid_features = process_features(valid_features)
        
        for idx in range(len(train_features)) :
            try :
                train_centers = np.array([np.mean(train_features[idx][train_labels==i], axis=0)/np.linalg.norm(np.mean(train_features[idx][train_labels==i], axis=0)) for i in range(np.max(train_labels)+1)])
                valid_centers = np.array([np.mean(valid_features[idx][valid_labels==i], axis=0)/np.linalg.norm(np.mean(valid_features[idx][valid_labels==i], axis=0)) for i in range(np.max(valid_labels)+1)])
                plot_weight_and_centers(centers=train_centers, weight=centers_logits, model_name=getattr(self.args, 'model_name', None), save_path=os.path.join(self.save_path,f'training_{feature_names[idx]}_centers_based_weight.png'))
                plot_weight_and_centers(centers=valid_centers, weight=centers_logits, model_name=getattr(self.args, 'model_name', None), save_path=os.path.join(self.save_path,f'valid_{feature_names[idx]}_centers_based_weight.png'))
                train_center_angles = np.arccos(train_centers@train_centers.T) * 180.0 / np.pi
                valid_center_angles = np.arccos(valid_centers@valid_centers.T) * 180.0 / np.pi
                model_names = [getattr(self.args, 'model_name', None)] if hasattr(self.args, 'model_name') else None
                plot_angle_matrix(angle_matrices=train_center_angles, save_path=os.path.join(self.save_path,f'training_{feature_names[idx]}_centers.png'), model_names=model_names, dataset_name=self.args.dataset_name)
                plot_angle_matrix(angle_matrices=valid_center_angles, save_path=os.path.join(self.save_path,f'valid_{feature_names[idx]}_centers.png'), model_names=model_names, dataset_name=self.args.dataset_name)
                plot_feature_distribution(features=valid_features[idx], labels=valid_labels, W=centers_logits, train=False, save_path=os.path.join(self.save_path,f'valid_{feature_names[idx]}_feature_distribution'))
                plot_feature_distribution(features=train_features[idx], labels=train_labels, W=centers_logits, train=True, save_path=os.path.join(self.save_path,f'training_{feature_names[idx]}_feature_distribution'))
            except Exception as e:
                print(f'Error : {e} \n {feature_names[idx]}')
                continue

    def main(self):
        self.analyze_dataset()
        if self.args.mode == 'dataset':
            return 
        self.analyze_model_performance()
        #self.analyze_features()
        #self.backbone_analysis.main()


class Compare:
    def __init__(self, args):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        train_valid_sets = load_dataset(args.dataset_path, dataset_name=args.dataset_name, imb_factor=args.imb_factor)
        self.train_set = train_valid_sets[0]
        self.valid_set = train_valid_sets[1]
        self.train_loader, self.valid_loader = load_loaders(self.train_set, self.valid_set)
        self.models = [get_model(args) for _ in range(len(args.model_paths))]
        self.model_names = args.model_names
        for model, model_path in zip(self.models, args.model_paths):
            model.load_state_dict(torch.load(model_path, map_location=self.device, weights_only=False)['model_state_dict'])
        self.aligner = load_aligner(args.aligner_path) if args.aligner_path else None
        self.save_path = args.save_path
        if not os.path.exists(self.save_path):
            os.makedirs(self.save_path,exist_ok=True)

    def main(self):
        train_preds, train_labels, train_confs = [], [], []
        valid_preds, valid_labels, valid_confs = [], [], []
        for model in self.models :
            preds, labels, confs = get_predictions(model, self.train_loader, self.aligner)
            v_preds, v_labels, v_confs = get_predictions(model, self.valid_loader, self.aligner)
            train_preds.append(preds)
            train_labels.append(labels)
            train_confs.append(confs)
            valid_preds.append(v_preds)
            valid_labels.append(v_labels)
            valid_confs.append(v_confs)

        accuracies = []
        for preds, labels in zip(valid_preds, valid_labels):
            accuracies.append(compute_accuracy_per_class(preds=preds, labels=labels))


        plot_accuracy_comparison(accuracies, self.model_names, self.train_set.labels, self.save_path)

        angle_matrices = [compute_angle_matrix(model) for model in self.models]
        compare_angle_rates(angle_matrices, self.model_names, self.save_path)
    

class Analyze_backbone:
    def __init__(self, args):
        self.args = args
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        n_c = 100 if '100' in args.dataset_name else 10
        classifier = get_model(args)
        model_ckpt = os.path.join(args.model_paths[0], 'best_acc.pth')
        classifier.load_state_dict(torch.load(model_ckpt, map_location=self.device, weights_only=False)['model_state_dict'])
        self.model = classifier.backbone
        self.model.eval()
        self.model = self.model.to(self.device)
        self.save_path = args.save_path
        if not os.path.exists(self.save_path):
            os.makedirs(self.save_path,exist_ok=True)
        train_valid_sets = load_dataset(args.dataset_path, dataset_name=args.dataset_name, imb_factor=args.imb_factor)
        self.train_set = train_valid_sets[0]
        self.valid_set = train_valid_sets[1]
        self.train_loader, self.valid_loader = load_loaders(self.train_set, self.valid_set)

    def main(self):
        features, labels = get_features_from_backbone(self.model, self.valid_loader)
        centers = get_centers(features, labels) # n_c , dim 
        center_angles = np.arccos(centers@centers.T) * 180.0 / np.pi
        plot_angle_matrix(center_angles, os.path.join(self.save_path,'centers.png'), model_names=self.args.model_names, dataset_name=self.args.dataset_name)

