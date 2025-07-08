#!/usr/bin/env python3
"""
YOLO Training Script for Person Detection
"""

import os
import sys
import yaml
from pathlib import Path
from ultralytics import YOLO
import torch


class YOLOTrainer:
    def __init__(self, model_name='yolov8n.pt', data_config='person_data.yaml'):
        """
        Initialize YOLO trainer

        Args:
            model_name: YOLO model to use (yolov8n.pt, yolov8s.pt, yolov8m.pt, yolov8l.pt, yolov8x.pt)
            data_config: Path to data configuration file
        """
        self.model_name = model_name
        self.data_config = data_config
        self.model = None

    def setup_model(self):
        """Setup YOLO model"""
        print(f"Loading YOLO model: {self.model_name}")
        self.model = YOLO(self.model_name)

    def create_data_config(self, train_path, val_path, test_path=None):
        """
        Create data configuration file for YOLO training

        Args:
            train_path: Path to training images
            val_path: Path to validation images  
            test_path: Path to test images (optional)
        """
        data_config = {
            'train': train_path,
            'val': val_path,
            'nc': 1,  # number of classes (person only)
            'names': ['person']
        }

        if test_path:
            data_config['test'] = test_path

        with open(self.data_config, 'w') as f:
            yaml.dump(data_config, f)

        print(f"Data configuration saved to: {self.data_config}")

    def train(self, epochs=100, imgsz=640, batch_size=16, device='auto',
              save_dir='../../models/yolo', project_name='person_detection'):
        """
        Train YOLO model

        Args:
            epochs: Number of training epochs
            imgsz: Image size for training
            batch_size: Batch size
            device: Device to use ('auto', 'cpu', '0', '1', etc.)
            save_dir: Directory to save model checkpoints
            project_name: Project name for organizing runs
        """
        if self.model is None:
            self.setup_model()

        # Create save directory
        os.makedirs(save_dir, exist_ok=True)

        print(f"Starting training with the following parameters:")
        print(f"  - Epochs: {epochs}")
        print(f"  - Image size: {imgsz}")
        print(f"  - Batch size: {batch_size}")
        print(f"  - Device: {device}")
        print(f"  - Save directory: {save_dir}")

        # Train the model
        results = self.model.train(
            data=self.data_config,
            epochs=epochs,
            imgsz=imgsz,
            batch=batch_size,
            device=device,
            project=save_dir,
            name=project_name,
            save=True,
            save_period=10,  # Save checkpoint every 10 epochs
            val=True,
            plots=True,
            verbose=True
        )

        return results

    def validate(self, model_path=None, data_config=None):
        """
        Validate trained model

        Args:
            model_path: Path to trained model (if None, uses current model)
            data_config: Data configuration file (if None, uses current config)
        """
        if model_path:
            self.model = YOLO(model_path)
        elif self.model is None:
            raise ValueError(
                "No model loaded. Please provide model_path or train a model first.")

        if data_config:
            self.data_config = data_config

        print("Validating model...")
        results = self.model.val(data=self.data_config)

        return results


def main():
    """Main training function"""
    import argparse

    parser = argparse.ArgumentParser(
        description='Train YOLO for person detection')
    parser.add_argument('--model', default='yolov8n.pt',
                        help='YOLO model to use')
    parser.add_argument('--data', default='person_data.yaml',
                        help='Data configuration file')
    parser.add_argument('--epochs', type=int, default=100,
                        help='Number of training epochs')
    parser.add_argument('--imgsz', type=int, default=640, help='Image size')
    parser.add_argument('--batch', type=int, default=16, help='Batch size')
    parser.add_argument('--device', default='auto', help='Device to use')
    parser.add_argument('--train-path', required=True,
                        help='Path to training images')
    parser.add_argument('--val-path', required=True,
                        help='Path to validation images')
    parser.add_argument('--test-path', help='Path to test images')
    parser.add_argument('--save-dir', default='../../models/yolo',
                        help='Directory to save models')

    args = parser.parse_args()

    # Initialize trainer
    trainer = YOLOTrainer(model_name=args.model, data_config=args.data)

    # Create data configuration
    trainer.create_data_config(
        train_path=args.train_path,
        val_path=args.val_path,
        test_path=args.test_path
    )

    # Train model
    results = trainer.train(
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch_size=args.batch,
        device=args.device,
        save_dir=args.save_dir
    )

    print("Training completed!")
    print(f"Results: {results}")


if __name__ == "__main__":
    main()
