# debug_model.py
import sys
import numpy as np
from pathlib import Path

sys.path.append('.')

from utils.data_loader import DataLoader
from models.keras_model import KerasModel
from sklearn.metrics import classification_report

print("Loading dataset...")
data_loader = DataLoader(dataset_path="Dataset")

# Load a small dataset for testing
X, y, y_action, paths, action_names = data_loader.prepare_dataset(
    max_videos_per_class=2,  # Just 2 videos per class for quick test
    feature_type='aggregated'
)

print(f"\nDataset shape: {X.shape}")
print(f"Labels: {y}")
print(f"Class distribution: {np.bincount(y)}")

if X is not None and len(X) > 0:
    # Split data
    X_train, X_val, X_test, y_train, y_val, y_test = data_loader.split_data(
        X, y, test_size=0.2, val_size=0.1
    )
    
    print(f"\nTrain: {X_train.shape}, Test: {X_test.shape}")
    
    # Create a simple model
    input_shape = (X_train.shape[1],)
    model = KerasModel(input_shape, num_classes=3, name="Test_Model", architecture='medium')
    model.build_model()
    
    print("\nTraining model...")
    history = model.train(
        X_train, y_train,
        X_val, y_val,
        epochs=10,
        batch_size=4,
        verbose=1
    )
    
    # Evaluate
    print("\nEvaluating model...")
    metrics = model.evaluate(X_test, y_test)
    
    print("\n=== Results ===")
    print(f"Accuracy: {metrics['accuracy']:.4f}")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall: {metrics['recall']:.4f}")
    print(f"F1-Score: {metrics['f1_score']:.4f}")
    
    # Get predictions
    y_pred = model.predict(X_test)
    
    # Print classification report
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=['Bad', 'Moderate', 'Good']))
    
    # Show which videos were misclassified
    print("\nMisclassified samples:")
    for i in range(len(X_test)):
        if y_test[i] != y_pred[i]:
            print(f"  Video: {paths[i]}")
            print(f"    True: {['Bad', 'Moderate', 'Good'][y_test[i]]}")
            print(f"    Pred: {['Bad', 'Moderate', 'Good'][y_pred[i]]}")
            print()
else:
    print("No data loaded! Check your dataset path and structure.")