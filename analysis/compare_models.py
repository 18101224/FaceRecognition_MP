import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
import torch
from typing import List, Dict, Tuple

def calculate_macro_accuracy(targets: torch.Tensor, predictions: torch.Tensor, 
                           class_counts: Dict[int, int]) -> float:
    """
    Calculate macro accuracy across all classes
    
    Args:
        targets (torch.Tensor): Ground truth labels
        predictions (torch.Tensor): Model predictions
        class_counts (Dict[int, int]): Number of samples per class
        
    Returns:
        float: Macro accuracy
    """
    # Convert to numpy for easier manipulation
    targets = targets.cpu().numpy()
    predictions = predictions.cpu().numpy()
    
    # Calculate per-class accuracy
    class_accuracies = {}
    for cls in class_counts.keys():
        # Get indices where true label is current class
        cls_indices = np.where(targets == cls)[0]
        if len(cls_indices) == 0:
            continue
            
        # Calculate accuracy for this class
        correct = np.sum(predictions[cls_indices] == cls)
        class_accuracies[cls] = correct / len(cls_indices)
    
    # Calculate macro accuracy (average of per-class accuracies)
    macro_accuracy = np.mean(list(class_accuracies.values()))
    return macro_accuracy

def plot_error_histogram(targets: torch.Tensor, 
                        predictions1: torch.Tensor, 
                        predictions2: torch.Tensor,
                        model1_name: str = "Model 1",
                        model2_name: str = "Model 2",
                        output_filename: str = "error_comparison.png"):
    """
    Plot histogram comparing errors between two models
    
    Args:
        targets (torch.Tensor): Ground truth labels
        predictions1 (torch.Tensor): Predictions from first model
        predictions2 (torch.Tensor): Predictions from second model
        model1_name (str): Name of first model
        model2_name (str): Name of second model
        output_filename (str): Output file name for the plot
    """
    # Convert to numpy
    targets = targets.cpu().numpy()
    predictions1 = predictions1.cpu().numpy()
    predictions2 = predictions2.cpu().numpy()
    
    # Calculate errors for each model
    errors1 = predictions1 != targets
    errors2 = predictions2 != targets
    
    # Count errors per class
    error_counts1 = {}
    error_counts2 = {}
    for cls in np.unique(targets):
        cls_indices = np.where(targets == cls)[0]
        error_counts1[cls] = np.sum(errors1[cls_indices])
        error_counts2[cls] = np.sum(errors2[cls_indices])
    
    # Create figure
    plt.figure(figsize=(12, 6))
    
    # Plot error counts
    x = np.arange(len(error_counts1))
    width = 0.35
    
    plt.bar(x - width/2, list(error_counts1.values()), width, label=model1_name, color='skyblue')
    plt.bar(x + width/2, list(error_counts2.values()), width, label=model2_name, color='lightcoral')
    
    plt.xlabel('Target Class')
    plt.ylabel('Number of Errors')
    plt.title('Error Comparison by Target Class')
    plt.xticks(x, list(error_counts1.keys()))
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(output_filename, dpi=300, bbox_inches='tight')
    plt.close()

def compare_models(targets: torch.Tensor,
                  predictions1: torch.Tensor,
                  predictions2: torch.Tensor,
                  class_counts: Dict[int, int],
                  model1_name: str = "Model 1",
                  model2_name: str = "Model 2",
                  output_prefix: str = "model_comparison"):
    """
    Compare two models by calculating metrics and plotting visualizations
    
    Args:
        targets (torch.Tensor): Ground truth labels
        predictions1 (torch.Tensor): Predictions from first model
        predictions2 (torch.Tensor): Predictions from second model
        class_counts (Dict[int, int]): Number of samples per class
        model1_name (str): Name of first model
        model2_name (str): Name of second model
        output_prefix (str): Prefix for output files
    """
    # Calculate macro accuracies
    macro_acc1 = calculate_macro_accuracy(targets, predictions1, class_counts)
    macro_acc2 = calculate_macro_accuracy(targets, predictions2, class_counts)
    
    # Print comparison
    print("\nModel Comparison:")
    print(f"{model1_name} Macro Accuracy: {macro_acc1:.4f}")
    print(f"{model2_name} Macro Accuracy: {macro_acc2:.4f}")
    print(f"Difference: {abs(macro_acc1 - macro_acc2):.4f}")
    
    # Plot error comparison
    error_plot_path = f"{output_prefix}_error_comparison.png"
    plot_error_histogram(targets, predictions1, predictions2, 
                        model1_name, model2_name, error_plot_path)
    print(f"\nError comparison plot saved to {error_plot_path}")
    
    # Print per-class accuracies
    print("\nPer-class Accuracies:")
    print(f"{'Class':<10} {model1_name:<15} {model2_name:<15}")
    print("-" * 40)
    
    for cls in sorted(class_counts.keys()):
        cls_indices = np.where(targets.cpu().numpy() == cls)[0]
        acc1 = np.mean(predictions1.cpu().numpy()[cls_indices] == cls)
        acc2 = np.mean(predictions2.cpu().numpy()[cls_indices] == cls)
        print(f"{cls:<10} {acc1:.4f}          {acc2:.4f}")

def main():
    # Example usage
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--targets_path', type=str, required=True, help='Path to target labels')
    parser.add_argument('--predictions1_path', type=str, required=True, help='Path to first model predictions')
    parser.add_argument('--predictions2_path', type=str, required=True, help='Path to second model predictions')
    parser.add_argument('--class_counts_path', type=str, required=True, help='Path to class counts')
    parser.add_argument('--model1_name', type=str, default='Model 1', help='Name of first model')
    parser.add_argument('--model2_name', type=str, default='Model 2', help='Name of second model')
    parser.add_argument('--output_prefix', type=str, default='model_comparison', help='Prefix for output files')
    args = parser.parse_args()
    
    # Load data
    targets = torch.load(args.targets_path)
    predictions1 = torch.load(args.predictions1_path)
    predictions2 = torch.load(args.predictions2_path)
    class_counts = torch.load(args.class_counts_path)
    
    # Compare models
    compare_models(targets, predictions1, predictions2, class_counts,
                  args.model1_name, args.model2_name, args.output_prefix)

if __name__ == "__main__":
    main() 