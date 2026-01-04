from ultralytics import YOLO
import os

def train_yolo(data_yaml_path, epochs=50, imgsz=416, device=0, batch=16):
    # Load a model
    model = YOLO("yolov8s.pt")  # load a pretrained model (recommended for training)

    # Train the model
    results = model.train(
        data=data_yaml_path,
        epochs=epochs,
        imgsz=imgsz,
        device=device,
        batch=batch,
        patience=10, # Early stopping if no improvement for 10 epochs
        save=True,
        project=os.path.join(os.path.dirname(data_yaml_path), 'runs/detect'),
        name='train',
        exist_ok=True # Overwrite existing experiment name
    )
    
    print("Training Complete!")
    print(f"Best model saved at: {os.path.join(model.trainer.save_dir, 'weights/best.pt')}")

if __name__ == "__main__":
    # Define paths
    base_dir = r"d:\Personal\Thesis\ASL-to-Voice-web"
    data_yaml = os.path.join(base_dir, "combined_dataset", "data.yaml")
    
    # Ensure data.yaml exists
    if not os.path.exists(data_yaml):
        print(f"Error: {data_yaml} not found!")
        exit(1)
        
    print(f"Starting training with data: {data_yaml}")
    train_yolo(data_yaml)
