from osgeo import gdal
import numpy as np
import os

def split_raster(input_tif, output_dir, tile_size=128):
    os.makedirs(output_dir, exist_ok=True)
    dataset = gdal.Open(input_tif)
    if dataset is None:
        raise FileNotFoundError("入力ファイルが見つかりません。")
    
    xsize, ysize = dataset.RasterXSize, dataset.RasterYSize
    num_bands = dataset.RasterCount
    region_name = os.path.basename(input_tif).replace("mask_", "").replace("_clip.tif", "")
    
    for i in range(0, xsize, tile_size):
        for j in range(0, ysize, tile_size):
            width = min(tile_size, xsize - i)
            height = min(tile_size, ysize - j)
            
            tile_data = np.zeros((num_bands, height, width), dtype=np.float32)
            for b in range(num_bands):
                band = dataset.GetRasterBand(b + 1)
                tile_data[b] = band.ReadAsArray(i, j, width, height)
            
            output_path = os.path.join(output_dir, f"tile_{region_name}_{i:05d}_{j:05d}.tif")
            driver = gdal.GetDriverByName("GTiff")
            out_ds = driver.Create(output_path, width, height, num_bands, gdal.GDT_Float32)
            out_ds.SetGeoTransform((
                dataset.GetGeoTransform()[0] + i * dataset.GetGeoTransform()[1], dataset.GetGeoTransform()[1], 0,
                dataset.GetGeoTransform()[3] + j * dataset.GetGeoTransform()[5], 0, dataset.GetGeoTransform()[5]
            ))
            out_ds.SetProjection(dataset.GetProjection())
            
            for b in range(num_bands):
                out_ds.GetRasterBand(b + 1).WriteArray(tile_data[b])
            
            out_ds = None  # 保存
            print(f"{output_path} を保存しました。")
    
    print("タイル分割が完了しました！")

# 複数の地域をリストで指定
regions = ["gofukuya","iburi","notoRain", "notoEarth"]  # 地域名のリスト（必要に応じて追加）

# 出力ディレクトリ
output_dir = "/media/hina/AE60-1503/sabo_UNet_Planet_trial03/tiles/masks"

# 各地域に対してタイル分割実行
for region in regions:
    input_tif = f"/media/hina/AE60-1503/sabo_UNet_Planet_trial03/masks/mask_{region}_clip.tif"
    split_raster(input_tif, output_dir, tile_size=128)
