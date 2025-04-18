import os
import time
from osgeo import gdal
import numpy as np

base_dir = "/media/hina/AE60-1503/sabo_UNet_Planet_trial03/tiles/"
masks_dir = os.path.join(base_dir, "masks")
images_dir = os.path.join(base_dir, "features")

deleted_files = []

file_list = [f for f in os.listdir(masks_dir) if f.lower().endswith(".tif")]

start = time.time()

for i, filename in enumerate(file_list):
    print(f"{i+1}/{len(file_list)} 処理中: {filename}")

    mask_path = os.path.join(masks_dir, filename)
    image_path = os.path.join(images_dir, filename)

    dataset = gdal.Open(mask_path)
    if dataset is None:
        print(f"  ⚠️ 開けませんでした: {filename}")
        continue

    band = dataset.GetRasterBand(1)
    array = band.ReadAsArray()
    total_pixels = array.size
    count_2 = np.sum(array == 1)
    ratio = count_2 / total_pixels

    if ratio > 0.85:
        if os.path.exists(mask_path):
            os.remove(mask_path)
        if os.path.exists(image_path):
            os.remove(image_path)
        deleted_files.append(filename)

elapsed = time.time() - start

print(f"\n✅ 削除されたペア数: {len(deleted_files)}（処理時間: {elapsed:.2f}秒）")
for f in deleted_files:
    print(f" - {f}")
