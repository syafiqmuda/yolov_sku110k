from ultralytics import YOLO, checks, hub

checks()

hub.login('40815389e96288255fd1121f17b108fbf9cedc7421')
model = YOLO('https://hub.ultralytics.com/models/aMnqOTD7cqVO1RJ2s1tC')
results = model.train()