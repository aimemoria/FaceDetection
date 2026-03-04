#!/usr/bin/env python3
"""
E. Training Pipeline — Stage A Person Detection

Trains Stage A: Person vs No-Person (binary classifier).

Features:
- Early stopping and learning rate reduction
- Class weighting for imbalanced data
- Confusion matrix and accuracy metrics
- Model checkpointing

Usage:
    python E_train_model.py --data_dir processed --epochs 100
"""

import os
import json
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import callbacks
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
from pathlib import Path
import argparse
from typing import Tuple, Dict, List

# Import model architectures
from D_model_architecture import create_stage_a_model, print_model_analysis

# =============================================================================
# Configuration
# =============================================================================
BATCH_SIZE = 32
EPOCHS = 150
LEARNING_RATE = 0.0005
EARLY_STOPPING_PATIENCE = 25
LR_REDUCE_PATIENCE = 12
LR_REDUCE_FACTOR = 0.5
MIN_LR = 1e-6


# =============================================================================
# Data Loading
# =============================================================================
def load_data(data_dir: str, stage: str) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
    """
    Load preprocessed data for a specific stage.
    
    Args:
        data_dir: Directory with processed NPZ files
        stage: 'a' for Stage A
        
    Returns:
        Dictionary with 'train', 'val', 'test' tuples
    """
    data_path = Path(data_dir)
    
    prefix = f'stage_{stage}'
    
    train_data = np.load(data_path / f'{prefix}_train.npz')
    val_data = np.load(data_path / f'{prefix}_val.npz')
    test_data = np.load(data_path / f'{prefix}_test.npz')
    
    return {
        'train': (train_data['X'], train_data['y']),
        'val': (val_data['X'], val_data['y']),
        'test': (test_data['X'], test_data['y'])
    }


def compute_class_weights(y: np.ndarray) -> Dict[int, float]:
    """
    Compute class weights for imbalanced datasets using squared inverse
    frequency weighting, capped at 10x to avoid training instability.

    Args:
        y: Array of class labels

    Returns:
        Dictionary mapping class index to weight
    """
    classes = np.unique(y)
    counts = {cls: int(np.sum(y == cls)) for cls in classes}
    max_count = max(counts.values())

    weights = {}
    for cls in classes:
        raw_weight = (max_count / counts[cls]) ** 1.5
        weights[cls] = min(raw_weight, 10.0)

    return weights



# =============================================================================
# Callbacks
# =============================================================================
def create_callbacks(model_name: str, 
                     output_dir: str) -> List[callbacks.Callback]:
    """
    Create training callbacks.
    
    Args:
        model_name: Name for saving checkpoints
        output_dir: Directory to save models
        
    Returns:
        List of Keras callbacks
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    callback_list = [
        # Model checkpoint - save best model
        callbacks.ModelCheckpoint(
            filepath=str(output_path / f'{model_name}_best.keras'),
            monitor='val_accuracy',
            mode='max',
            save_best_only=True,
            verbose=1
        ),
        
        # Early stopping
        callbacks.EarlyStopping(
            monitor='val_accuracy',
            patience=EARLY_STOPPING_PATIENCE,
            mode='max',
            restore_best_weights=True,
            verbose=1
        ),
        
        # Reduce learning rate on plateau
        callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=LR_REDUCE_FACTOR,
            patience=LR_REDUCE_PATIENCE,
            min_lr=MIN_LR,
            verbose=1
        ),
        
        # TensorBoard logging
        callbacks.TensorBoard(
            log_dir=str(output_path / 'logs' / model_name),
            histogram_freq=0,
            write_graph=True
        )
    ]
    
    return callback_list


# =============================================================================
# Training Functions
# =============================================================================
def train_model(model: keras.Model,
                train_data: Tuple[np.ndarray, np.ndarray],
                val_data: Tuple[np.ndarray, np.ndarray],
                model_name: str,
                output_dir: str,
                epochs: int = EPOCHS,
                batch_size: int = BATCH_SIZE,
                use_class_weights: bool = True) -> keras.callbacks.History:
    """
    Train a model with the given data.
    
    Args:
        model: Keras model to train
        train_data: (X_train, y_train) tuple
        val_data: (X_val, y_val) tuple
        model_name: Name for saving
        output_dir: Output directory
        epochs: Number of epochs
        batch_size: Batch size
        use_class_weights: Whether to use class weighting
        
    Returns:
        Training history
    """
    X_train, y_train = train_data
    X_val, y_val = val_data
    
    # Compile model
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=LEARNING_RATE),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    # Compute class weights
    class_weights = None
    if use_class_weights:
        class_weights = compute_class_weights(y_train)
        print(f"\nClass weights: {class_weights}")
    
    # Create callbacks
    callback_list = create_callbacks(model_name, output_dir)
    
    # Print training info
    print(f"\n{'='*60}")
    print(f"Training {model_name}")
    print(f"{'='*60}")
    print(f"Training samples: {len(X_train)}")
    print(f"Validation samples: {len(X_val)}")
    print(f"Batch size: {batch_size}")
    print(f"Max epochs: {epochs}")
    
    # Train
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        class_weight=class_weights,
        callbacks=callback_list,
        verbose=1
    )
    
    return history


# =============================================================================
# Evaluation Functions
# =============================================================================
def evaluate_model(model: keras.Model,
                   test_data: Tuple[np.ndarray, np.ndarray],
                   class_names: List[str],
                   model_name: str,
                   output_dir: str):
    """
    Evaluate model on test data and generate reports.
    
    Args:
        model: Trained Keras model
        test_data: (X_test, y_test) tuple
        class_names: List of class names
        model_name: Model name for saving
        output_dir: Output directory
    """
    X_test, y_test = test_data
    
    # Get predictions
    y_pred_probs = model.predict(X_test, verbose=0)
    y_pred = np.argmax(y_pred_probs, axis=1)
    
    # Calculate accuracy
    accuracy = np.mean(y_pred == y_test)
    
    print(f"\n{'='*60}")
    print(f"Evaluation Results: {model_name}")
    print(f"{'='*60}")
    print(f"Test Accuracy: {accuracy*100:.2f}%")
    
    # Classification report
    print(f"\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=class_names))
    
    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    print(f"\nConfusion Matrix:")
    print(cm)
    
    # Save confusion matrix plot
    output_path = Path(output_dir)
    plot_confusion_matrix(cm, class_names, 
                          str(output_path / f'{model_name}_confusion_matrix.png'))
    
    # Save metrics to JSON
    metrics = {
        'model_name': model_name,
        'test_accuracy': float(accuracy),
        'confusion_matrix': cm.tolist(),
        'class_names': class_names
    }
    
    with open(output_path / f'{model_name}_metrics.json', 'w') as f:
        json.dump(metrics, f, indent=2)
    
    return accuracy, cm


def plot_confusion_matrix(cm: np.ndarray, 
                          class_names: List[str],
                          save_path: str):
    """Plot and save confusion matrix."""
    fig, ax = plt.subplots(figsize=(8, 6))
    
    im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax)
    
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           xticklabels=class_names,
           yticklabels=class_names,
           title='Confusion Matrix',
           ylabel='True label',
           xlabel='Predicted label')
    
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")
    
    # Add text annotations
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], 'd'),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    
    fig.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Confusion matrix saved to: {save_path}")


def plot_training_history(history: keras.callbacks.History,
                          model_name: str,
                          save_path: str):
    """Plot training history (accuracy and loss)."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    
    # Accuracy plot
    axes[0].plot(history.history['accuracy'], label='Train')
    axes[0].plot(history.history['val_accuracy'], label='Validation')
    axes[0].set_title(f'{model_name} - Accuracy')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Accuracy')
    axes[0].legend()
    axes[0].grid(True)
    
    # Loss plot
    axes[1].plot(history.history['loss'], label='Train')
    axes[1].plot(history.history['val_loss'], label='Validation')
    axes[1].set_title(f'{model_name} - Loss')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Loss')
    axes[1].legend()
    axes[1].grid(True)
    
    fig.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Training history saved to: {save_path}")



# =============================================================================
# Main Training Pipeline
# =============================================================================
def main():
    parser = argparse.ArgumentParser(description='Train face detection model')
    parser.add_argument('--data_dir', type=str, default='processed',
                        help='Directory with preprocessed data')
    parser.add_argument('--output_dir', type=str, default='models',
                        help='Directory to save trained models')
    parser.add_argument('--epochs', type=int, default=EPOCHS,
                        help='Maximum training epochs')
    parser.add_argument('--batch_size', type=int, default=BATCH_SIZE,
                        help='Training batch size')
    args = parser.parse_args()
    
    # Set random seeds
    np.random.seed(42)
    tf.random.set_seed(42)
    
    # Load metadata
    data_path = Path(args.data_dir)
    with open(data_path / 'metadata.json', 'r') as f:
        metadata = json.load(f)
    
    print("=" * 60)
    print("TinyML Face Detection Training Pipeline")
    print("=" * 60)
    print(f"\nData directory: {args.data_dir}")
    print(f"Output directory: {args.output_dir}")
    print(f"Image size: {metadata['img_size']}x{metadata['img_size']}")
    
    output_path = Path(args.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    print("\n" + "=" * 60)
    print("STAGE A: Person Detection Training")
    print("=" * 60)

    stage_a_data = load_data(args.data_dir, 'a')
    class_names_a = metadata['class_names_stage_a']

    print(f"\nClasses: {class_names_a}")
    print(f"Train: {len(stage_a_data['train'][0])}")
    print(f"Val:   {len(stage_a_data['val'][0])}")
    print(f"Test:  {len(stage_a_data['test'][0])}")

    model_a = create_stage_a_model()
    print_model_analysis(model_a)

    history_a = train_model(
        model_a,
        stage_a_data['train'],
        stage_a_data['val'],
        'stage_a',
        args.output_dir,
        epochs=args.epochs,
        batch_size=args.batch_size
    )

    plot_training_history(history_a, 'Stage A',
                          str(output_path / 'stage_a_history.png'))

    acc_a, cm_a = evaluate_model(
        model_a,
        stage_a_data['test'],
        class_names_a,
        'stage_a',
        args.output_dir
    )

    model_a.save(output_path / 'stage_a_final.keras')
    print(f"\nStage A model saved to: {output_path / 'stage_a_final.keras'}")

    with open(output_path / 'training_metadata.json', 'w') as f:
        json.dump(metadata, f, indent=2)

    print("\n" + "=" * 60)
    print("Training Complete!")
    print("=" * 60)
    print(f"\nNext step: python3 F_quantize_model.py --model_dir {args.output_dir}")


if __name__ == '__main__':
    main()
