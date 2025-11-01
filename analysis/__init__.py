from .computation import *
from .plot import *
from .load import *
import torch, os
import sys; sys.path.append('..')

feature_names = ['backbone_feat', 'cls_feat', 'bcl_feat', 'center_feat']

class Analysis:
    def __init__(self,args):
        self.args = args     
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.datasets = load_dataset(self.args,dataset_path=self.args.dataset_path, dataset_name=self.args.dataset_name, imb_factor=self.args.imb_factor)
        self.loaders = load_loaders(self.datasets)
        self.save_path = self.args.save_path
        if not os.path.exists(self.save_path):
            os.makedirs(self.save_path,exist_ok=True)

        if self.args.mode == 'dataset':
            return 
        log = load_logs(self.args.model_paths)[0]
        self.args = concat_args(self.args, [log])[0]

        #load dataset 


        #load model

        self.model = load_models(self.args.model_paths, [self.args])[0]
        self.aligner = load_aligner(self.args.aligner_path) if self.args.aligner_path else None
        # self.backbone_analysis = Analyze_backbone(self.args)
        self.backbone = self.model.backbone 
        self.weight = self.model.weight.T.detach().cpu().numpy()

    def analyze_dataset(self):
        plot_label_distribution(self.datasets[0].labels, self.datasets[1].labels, self.save_path, model_name=getattr(self.args, 'model_name', None),
        dataset_name=self.args.dataset_name, boundaries=self.datasets[0].boundaries)
        self.dataset_categories = self.datasets[0].get_macro_category()
        
    def analyze_model_performance(self):
        train_preds, train_labels, train_logits, train_features, train_features_branch, _, training_indices= get_predictions(self.model, self.loaders[0], self.aligner,get_features=True)
        valid_preds, valid_labels, valid_logits, valid_features, valid_features_branch, centers_af_branch, valid_indices = get_predictions(self.model, self.loaders[1], self.aligner,get_features=True)
        valid_preds_balanced, valid_labels_balanced, valid_confs_balanced, _, _, _, _ = get_predictions(self.model, self.loaders[2], self.aligner,get_features=True) if len(self.loaders) == 3 else (None, None, None, None, None, None, None)
        centers_wo_branch = self.model.get_kernel().T.detach().cpu().numpy()
        plot_dist(training_features=train_features, validation_features=valid_features, target_centers=centers_wo_branch,
         training_logits=train_logits,validation_logits=valid_logits,training_labels=train_labels,validation_labels=valid_labels,save_path=f'{self.save_path}/dist.png', log_scale=True)

        valid_prediction_indices, training_k_indices, training_k_dists = find_nearest_training_for_misclassified(training_features=train_features, validation_features=valid_features, target_centers=centers_wo_branch,  validation_logits=valid_logits, validation_labels=valid_labels,k=3)
        valid_prediction_indices = valid_indices[valid_prediction_indices] # dataset index 
        print(f'# mis classified samples : {valid_prediction_indices.shape[0]}')
        training_k_indices = training_indices[training_k_indices]
        # for i in list(range(valid_indices.shape[0]//8))[:10]:
        #     plot_val_train_neighbors_from_datasets(validation_indices=valid_prediction_indices[i*8:(i+1)*8], training_k_indices=training_k_indices[i*8:(i+1)*8], distances=training_k_dists[i*8:(i+1)*8], 
        #     val_dataset=self.loaders[1].dataset , train_dataset=self.loaders[0].dataset , val_labels=valid_labels, train_labels=train_labels, save_path=f'{self.save_path}/val_train_neighbors_from_datasets_{i}.png',n=8)
        visualize_neural_collapse(X_tr=train_features, Z_tr=train_logits, y_tr=train_labels, X_va=valid_features, Z_va=valid_logits, y_va=valid_labels,
                 W=centers_wo_branch, savepath=f'{self.save_path}/wo_branch')
        print('wo_branch done')
        # w. branch version 
        visualize_neural_collapse(X_tr=train_features_branch, Z_tr=train_logits, y_tr=train_labels, X_va=valid_features_branch, Z_va=valid_logits, y_va=valid_labels,
                 W=centers_af_branch, savepath=f'{self.save_path}/w_branch')
        print('w_branch done')
        valid_macro_accuracy, valid_macro_acc_per_class = get_macro_accuracy(valid_preds, valid_labels)
        valid_macro_acc_per_category = get_macro_category(self.dataset_categories, valid_macro_acc_per_class)
        print('CATEGORY ACCURACY : \n')
        for category, acc in valid_macro_acc_per_category.items():
            print(f'{category} : {acc * 100:.2f}%')
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
        print(f"Validation Macro Accuracy: {valid_macro_accuracy * 100:.2f}%")

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
        self.logs = load_logs(args.model_paths)
        self.args = concat_args(args, self.logs)
        self.datasets = load_dataset(self.args[0],self.args[0].dataset_path, dataset_name=self.args[0].dataset_name, imb_factor=self.args[0].imb_factor)
        self.loaders = load_loaders(self.datasets)
        self.models = load_models(args.model_paths, self.args)
        self.model_names = args.model_names
        self.aligner = load_aligner(args.aligner_path) if args.aligner_path else None
        self.save_path = args.save_path
        if not os.path.exists(self.save_path):
            os.makedirs(self.save_path,exist_ok=True)

    def main(self):
        train_preds, train_labels, train_confs = [], [], []
        valid_preds, valid_labels, valid_confs = [], [], []
        if len(self.loaders) == 3 :
            valid_preds_balanced, valid_labels_balanced, valid_confs_balanced = [], [], []
        for idx, model in enumerate(self.models) :
            preds, labels, confs = list(get_predictions(model, self.loaders[0], self.aligner))[:3]
            v_preds, v_labels, v_confs = list(get_predictions(model, self.loaders[1], self.aligner))[:3]
            if len(self.loaders) == 3 :
                v_preds_balanced, v_labels_balanced, v_confs_balanced = list(get_predictions(model, self.loaders[2], self.aligner))[:3]
                valid_preds_balanced.append(v_preds_balanced)
                valid_labels_balanced.append(v_labels_balanced)
                valid_confs_balanced.append(v_confs_balanced)
            train_preds.append(preds)
            train_labels.append(labels)
            train_confs.append(confs)
            valid_preds.append(v_preds)
            valid_labels.append(v_labels)
            valid_confs.append(v_confs)
            print(f'{self.args[0].model_names[idx]} Validation Macro Accuracy: {get_macro_accuracy(v_preds, v_labels)[0] * 100:.2f}%')

                
        valid_accuracies = []
        valid_accuracies_balanced = []
        for preds, labels  in zip(valid_preds, valid_labels):
            valid_accuracies.append(compute_accuracy_per_class(preds=preds, labels=labels))
        if len(self.loaders) == 3 :
            for preds, labels in zip(valid_preds_balanced, valid_labels_balanced):
                valid_accuracies_balanced.append(compute_accuracy_per_class(preds=preds, labels=labels))
        # (num_models, num_classes) * 2 

        plot_accuracy_comparison(valid_accuracies, self.model_names, self.datasets[0].labels, self.save_path)
        plot_accuracy_comparison(valid_accuracies_balanced, self.model_names, self.datasets[0].labels, self.save_path, save_name='valid_confusion_matrix_balanced.png') if len(self.loaders) == 3 else None

        angle_matrices = [compute_angle_matrix(model) for model in self.models]
        compare_angle_rates(angle_matrices, self.model_names, self.save_path)

        # Compare confusion matrices across models
        normed_cms = []
        normed_cms_balanced = [] if len(self.loaders) == 3 else None
        for idx, model in enumerate(self.models):
            v_preds, v_labels, _ = list(get_predictions(model, self.loaders[1], self.aligner))[:3]
            v_cm = compute_confusion_matrix(v_preds, v_labels)
            v_cm_norm = normalize_confusion_matrix(v_cm, v_labels)
            normed_cms.append(v_cm_norm)
            if len(self.loaders) == 3:
                vb_preds, vb_labels, _ = list(get_predictions(model, self.loaders[2], self.aligner))[:3]
                vb_cm = compute_confusion_matrix(vb_preds, vb_labels)
                vb_cm_norm = normalize_confusion_matrix(vb_cm, vb_labels)
                normed_cms_balanced.append(vb_cm_norm)

        compare_confusion_matrices(normed_cms, self.model_names, self.save_path, save_name='compare_confusion_matrices.png')
        if len(self.loaders) == 3:
            compare_confusion_matrices(normed_cms_balanced, self.model_names, self.save_path, save_name='compare_confusion_matrices_balanced.png')
        

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

