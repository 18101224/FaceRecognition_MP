import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc, precision_recall_curve, f1_score, precision_score, recall_score
from sklearn.preprocessing import label_binarize
import os
import argparse
from eval import EvalObject
from copy import deepcopy

def get_args():
    args = argparse.ArgumentParser()
    args.add_argument('--dataset_path', default=None, type=str)
    args.add_argument('--ckpt_path', default=None, type=str)
    args.add_argument('--eval_name', default=None, type=str)
    args.add_argument('--world_size',default=1, type=int)

    return args.parse_args()

class CosineClassifierAnalyzer:
    def __init__(self, args):
        self.eval_helper = EvalObject(args)
        if not os.path.exists(f'results/{args.eval_name}'):
            os.mkdir(f'results/{args.eval_name}')
        self.save_path = f'results/{args.eval_name}/'
    def get_class_centers(self):
        """Extract class centers (weights) from the cosine classifier."""
        # Assuming the last layer is the cosine classifier
        return self.eval_helper.model.classifier.kernel.detach().cpu().numpy().transpose()
    
    def compute_angle_matrix(self, centers):
        """Compute angle matrix between class centers."""
        num_classes = centers.shape[0]
        angle_matrix = np.zeros((num_classes, num_classes))
        
        for i in range(num_classes):
            for j in range(num_classes):
                # Compute cosine similarity
                cos_sim = np.dot(centers[i], centers[j]) / (np.linalg.norm(centers[i]) * np.linalg.norm(centers[j]))
                # Convert to angle in degrees
                angle = np.arccos(np.clip(cos_sim, -1.0, 1.0)) * 180 / np.pi
                angle_matrix[i, j] = angle
        
        return angle_matrix
    
    def plot_angle_matrix(self, angle_matrix, output_filename):
        """Plot angle matrix as a heatmap focusing only on inter-class angles."""
        # Create off-diagonal matrix
        off_diag_matrix = angle_matrix.copy()
        np.fill_diagonal(off_diag_matrix, np.nan)  # Set diagonal to NaN to ignore in color scaling
        
        plt.figure(figsize=(10, 8))
        sns.heatmap(off_diag_matrix,
                   annot=True,
                   fmt='.1f',
                   cmap='YlOrRd',
                   square=True,
                   vmin=np.nanmin(off_diag_matrix),  # Use min of off-diagonal elements
                   vmax=np.nanmax(off_diag_matrix),  # Use max of off-diagonal elements
                   cbar_kws={'label': 'Angle (degrees)'})
        plt.title('Inter-class Angles (degrees)')
        plt.xlabel('Class')
        plt.ylabel('Class')
        
        plt.tight_layout()
        plt.savefig(output_filename, dpi=300, bbox_inches='tight')
        plt.close()
    
    def plot_confusion_matrix(self, y_true, y_pred, output_filename):
        """Plot normalized confusion matrix focusing only on errors."""
        cm = confusion_matrix(y_true, y_pred)
        cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        
        # Create error matrix (off-diagonal elements only)
        error_matrix = cm_normalized.copy()
        np.fill_diagonal(error_matrix, np.nan)  # Set diagonal to NaN to ignore in color scaling
        
        plt.figure(figsize=(10, 8))
        sns.heatmap(error_matrix,
                   annot=True,
                   fmt='.3f',
                   cmap='Reds',
                   square=True,
                   vmin=np.nanmin(error_matrix),  # Use min of off-diagonal elements
                   vmax=np.nanmax(error_matrix),  # Use max of off-diagonal elements
                   cbar_kws={'label': 'Error Rate'})
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.title('Error Matrix (Off-diagonal elements)')
        
        plt.tight_layout()
        plt.savefig(output_filename, dpi=300, bbox_inches='tight')
        plt.close()

    def plot_metrics(self, valid_labels, valid_logits, valid_preds):
        # Calculate macro and micro averaged metrics
        macro_precision = precision_score(valid_labels, valid_preds, average='macro')
        macro_recall = recall_score(valid_labels, valid_preds, average='macro')
        macro_f1 = f1_score(valid_labels, valid_preds, average='macro')
        
        micro_precision = precision_score(valid_labels, valid_preds, average='micro')
        micro_recall = recall_score(valid_labels, valid_preds, average='micro')
        micro_f1 = f1_score(valid_labels, valid_preds, average='micro')
        
        # Create a figure with three subplots
        plt.figure(figsize=(15, 5))
        
        # Macro-averaged metrics
        plt.subplot(1, 3, 1)
        metrics = ['Precision', 'Recall', 'F1-Score']
        values = [macro_precision, macro_recall, macro_f1]
        plt.bar(metrics, values, color=['blue', 'green', 'red'])
        plt.ylim([0, 1])
        plt.title('Macro-averaged Metrics')
        for i, v in enumerate(values):
            plt.text(i, v + 0.02, f'{v:.4f}', ha='center')
        
        # Micro-averaged metrics
        plt.subplot(1, 3, 2)
        values = [micro_precision, micro_recall, micro_f1]
        plt.bar(metrics, values, color=['blue', 'green', 'red'])
        plt.ylim([0, 1])
        plt.title('Micro-averaged Metrics')
        for i, v in enumerate(values):
            plt.text(i, v + 0.02, f'{v:.4f}', ha='center')
        
        # Print metrics to console
        print("\nClassification Metrics:")
        print(f"Macro-averaged Precision: {macro_precision:.4f}")
        print(f"Macro-averaged Recall: {macro_recall:.4f}")
        print(f"Macro-averaged F1-Score: {macro_f1:.4f}")
        print(f"Micro-averaged Precision: {micro_precision:.4f}")
        print(f"Micro-averaged Recall: {micro_recall:.4f}")
        print(f"Micro-averaged F1-Score: {micro_f1:.4f}")
        
        plt.tight_layout()
        plt.savefig(f'{self.save_path}metrics.png', dpi=800)
        plt.close()

def main():
    args = get_args()
    
    # Initialize analyzer
    analyzer = CosineClassifierAnalyzer(args)
    
    # Get class centers and compute angle matrix
    print("Computing angle matrix between class centers...")
    centers = analyzer.get_class_centers()
    angle_matrix = analyzer.compute_angle_matrix(centers)
    
    # Plot angle matrix
    angle_matrix_path = f"{analyzer.save_path}"+"angle_matrix.png"
    analyzer.plot_angle_matrix(angle_matrix, angle_matrix_path)
    print(f"Angle matrix saved to {angle_matrix_path}")
    
    # Analyze predictions and plot confusion matrix
    print("Analyzing predictions...")
    train_acc, train_preds, train_labels, train_logits = analyzer.eval_helper.get_predictions(train=True)
    valid_acc, valid_preds, valid_labels, valid_logits = analyzer.eval_helper.get_predictions(train=False)
    
    # Plot confusion matrix
    confusion_matrix_path = analyzer.save_path + 'train_confusion_matrix.png'
    analyzer.plot_confusion_matrix(train_labels, train_preds, confusion_matrix_path)
    print(f"Confusion matrix saved to {confusion_matrix_path}")

    confusion_matrix_path = analyzer.save_path + 'valid_confusion_matrix.png'
    analyzer.plot_confusion_matrix(valid_labels, valid_preds, confusion_matrix_path)
    print(f"Confusion matrix saved to {confusion_matrix_path}")

    # Print angle statistics
    print("\nAngle Statistics:")
    print(f"Minimum angle between classes: {np.min(angle_matrix[angle_matrix > 0]):.2f}°")
    print(f"Maximum angle between classes: {np.max(angle_matrix):.2f}°")
    print(f"Mean angle between classes: {np.mean(angle_matrix[angle_matrix > 0]):.2f}°")

    print(f' training_acc :{train_acc:.4f}')
    print(f' validation_acc:{valid_acc:.4f}')

    # Plot metrics
    analyzer.plot_metrics(valid_labels, valid_logits, valid_preds)
    
if __name__ == "__main__":
    main() 