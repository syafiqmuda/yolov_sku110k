import os
from ultralytics import hub, checks

# Step 1: Use NVMe for downloads and cache
os.environ['ULTRALYTICS_CACHE_DIR'] = '/opt/dlami/nvme/ultralytics_cache'
os.makedirs(os.environ['ULTRALYTICS_CACHE_DIR'], exist_ok=True)

# Step 2: Optional: work in NVMe directory
os.chdir('/opt/dlami/nvme/yolov_sku110k')

checks()

hub.login('40815389e96288255fd1121f17b108fbf9cedc7421')

model = YOLO('https://hub.ultralytics.com/models/Mi50imfykoh6fdFR0z9r')
results = model.train()