from ultralytics import YOLO, checks, hub
checks()

hub.login('40815389e96288255fd1121f17b108fbf9cedc7421')

model = YOLO('yolo11s.pt')
results = model.train()