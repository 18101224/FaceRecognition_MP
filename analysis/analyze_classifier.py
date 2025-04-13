import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
from torchvision import transforms
from PIL import Image
import os
from collections import defaultdict

class ClassifierAnalyzer:
    def __init__(self, model_path, dataset_root):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = torch.load(model_path, map_location=self.device)
        self.model.eval()
        self.dataset_root = dataset_root
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
    def analyze_predictions(self, split='valid'):
        """Analyze model predictions on the specified split."""
        split_dir = os.path.join(self.dataset_root, split)
        all_preds = []
        all_labels = []
        class_correct = defaultdict(int)
        class_total = defaultdict(int)
        
        for cls in sorted(os.listdir(split_dir)):
            cls_dir = os.path.join(split_dir, cls)
            if not os.path.isdir(cls_dir):
                continue
                
            for img_name in os.listdir(cls_dir):
                img_path = os.path.join(cls_dir, img_name)
                try:
                    # Load and preprocess image
                    img = Image.open(img_path).convert('RGB')
                    img_tensor = self.transform(img).unsqueeze(0).to(self.device)
                    
                    # Get prediction
                    with torch.no_grad():
                        outputs = self.model(img_tensor)
                        _, predicted = torch.max(outputs, 1)
                        pred = predicted.item()
                    
                    # Update statistics
                    true_label = int(cls)
                    all_preds.append(pred)
                    all_labels.append(true_label)
                    
                    if pred == true_label:
                        class_correct[true_label] += 1
                    class_total[true_label] += 1
                    
                except Exception as e:
                    print(f"Error processing {img_path}: {e}")
        
        return all_preds, all_labels, class_correct, class_total
    
    def plot_confusion_matrix(self, y_true, y_pred):
        """Plot confusion matrix."""
        cm = confusion_matrix(y_true, y_pred)
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.title('Confusion Matrix')
        plt.savefig('confusion_matrix.png')
        plt.close()
        return cm

    def plot_class_accuracy(self, class_correct, class_total):
        """Plot accuracy per class."""
        accuracies = {cls: class_correct[cls] / class_total[cls] 
                     for cls in class_total.keys()}
        
        plt.figure(figsize=(10, 6))
        plt.bar(accuracies.keys(), accuracies.values())
        plt.xlabel('Class')
        plt.ylabel('Accuracy')
        plt.title('Accuracy per Class')
        plt.savefig('class_accuracy.png')
        plt.close()
    
    def analyze_feature_space(self, split='valid'):
        """Analyze the feature space of the model."""
        split_dir = os.path.join(self.dataset_root, split)
        features = []
        labels = []
        
        for cls in sorted(os.listdir(split_dir)):
            cls_dir = os.path.join(split_dir, cls)
            if not os.path.isdir(cls_dir):
                continue
                
            for img_name in os.listdir(cls_dir):
                img_path = os.path.join(cls_dir, img_name)
                try:
                    # Load and preprocess image
                    img = Image.open(img_path).convert('RGB')
                    img_tensor = self.transform(img).unsqueeze(0).to(self.device)
                    
                    # Get features (assuming the model has a feature extractor)
                    with torch.no_grad():
                        feature = self.model.feature_extractor(img_tensor)
                        features.append(feature.cpu().numpy().flatten())
                        labels.append(int(cls))
                    
                except Exception as e:
                    print(f"Error processing {img_path}: {e}")
        
        return np.array(features), np.array(labels)
    
    def plot_feature_space(self, features, labels):
        """Plot feature space using t-SNE."""
        from sklearn.manifold import TSNE
        
        # Reduce dimensionality for visualization
        tsne = TSNE(n_components=2, random_state=42)
        features_2d = tsne.fit_transform(features)
        
        plt.figure(figsize=(10, 8))
        scatter = plt.scatter(features_2d[:, 0], features_2d[:, 1], 
                            c=labels, cmap='tab10')
        plt.colorbar(scatter)
        plt.title('Feature Space Visualization (t-SNE)')
        plt.savefig('feature_space.png')
        plt.close()

def main():
    # Initialize analyzer
    analyzer = ClassifierAnalyzer(
        model_path='checkpoints/best_model.pth',  # Update with your model path
        dataset_root='dataset'  # Update with your dataset path
    )
    
    # Analyze predictions
    print("Analyzing predictions...")
    all_preds, all_labels, class_correct, class_total = analyzer.analyze_predictions()
    
    # Print classification report
    print("\nClassification Report:")
    print(classification_report(all_labels, all_preds))
    
    # Create visualizations
    print("\nCreating visualizations...")
    analyzer.plot_confusion_matrix(all_labels, all_preds)
    analyzer.plot_class_accuracy(class_correct, class_total)
    
    # Analyze feature space
    print("\nAnalyzing feature space...")
    features, labels = analyzer.analyze_feature_space()
    analyzer.plot_feature_space(features, labels)

if __name__ == "__main__":
    main() 