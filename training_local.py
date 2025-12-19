from ultralytics import YOLO, checks

checks()

if __name__ == "__main__":
    model = YOLO('yolo11s.pt')
    results = model.train(
        data="SKU-110K.yaml",
        imgsz = 720, # image quality (720p, 1280p, 1440p, 1920p)
        batch = 4, # total image proccesses at a time
        epochs = 100, # total model training (sees all images)
        workers = 4, # parallel data loading processes (workers x 2 = CPU)
        device=0, # total GPU
        patience=100,   # Stop early if no improvement
        augment=True, # Automatically applies transformations
        cache=False # if failed on continue
    )