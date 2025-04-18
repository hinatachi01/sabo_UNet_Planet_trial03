import os
import tifffile as tiff

# マスク画像だけを確認して、サイズ・バンド数が合わなければ両方削除
def remove_non_128_masks(masks_dir, images_dir):
    for fname in os.listdir(masks_dir):
        if fname.endswith('.tif'):
            mask_path = os.path.join(masks_dir, fname)
            img_path = os.path.join(images_dir, fname)

            try:
                mask = tiff.imread(mask_path)

                if mask.ndim == 2:
                    height, width = mask.shape
                    bands = 1
                elif mask.ndim == 3:
                    height, width, bands = mask.shape
                else:
                    print(f"{fname} has unexpected number of dimensions: {mask.ndim}")
                    continue

                print(f"{fname} has size: {height}x{width} pixels and {bands} bands.")

                if height != 128 or width != 128:
                    print(f"Removing: {fname}")
                    os.remove(mask_path)
                    if os.path.exists(img_path):
                        os.remove(img_path)

            except Exception as e:
                print(f"Error reading {fname}: {e}")

remove_non_128_masks(
    '/media/hina/AE60-1503/sabo_UNet_Planet_trial03/tiles/masks',
    '/media/hina/AE60-1503/sabo_UNet_Planet_trial03/tiles/features'
)
