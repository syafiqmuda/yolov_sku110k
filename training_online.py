from ultralytics import YOLO, checks, hub

checks()

if __name__ == "__main__":
    hub.login('40815389e96288255fd1121f17b108fbf9cedc7421')
    
    model = YOLO('https://hub.ultralytics.com/models/RDctHGy0iVk7Df4d5Ikc')
    results = model.train()