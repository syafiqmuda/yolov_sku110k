import os
from ultralytics import YOLO, checks, hub

os.environ['ULTRALYTICS_CACHE_DIR'] = '/opt/dlami/nvme/.ultralytics'

checks()

hub.login('40815389e96288255fd1121f17b108fbf9cedc7421')

model = YOLO('https://hub.ultralytics.com/models/XM6OQ0fDaHzFV7FEwuOW')
results = model.train()