import torch
from utils.model_analysis import plot_classifier_weights_tsne, analyze_classifier_weights
import os
from models import kprpe_fer

def main():
    # Create output directory if it doesn't exist
    os.makedirs('analysis_results', exist_ok=True)
    
    # Load your model
    # Replace these paths with your actual paths
    cfg_path = "path/to/your/config"
    checkpoint_path = "path/to/your/checkpoint"
    
    # Initialize model
    model = kprpe_fer(cfg_path)
    
    # Load checkpoint if exists
    if os.path.exists(checkpoint_path):
        model.load_from_state_dict(checkpoint_path)
    
    # Move model to CPU for analysis
    model = model.cpu()
    model.eval()
    
    # Analyze classifier weights
    print("Analyzing classifier weights...")
    analyze_classifier_weights(model)
    
    # Plot t-SNE visualization
    print("\nGenerating t-SNE visualization...")
    plot_classifier_weights_tsne(model, save_path='analysis_results/classifier_tsne.png')
    
    print("\nAnalysis complete! Results saved in 'analysis_results' directory")

if __name__ == "__main__":
    main() 