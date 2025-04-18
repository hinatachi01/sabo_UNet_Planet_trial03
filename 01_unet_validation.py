import os
import glob
import numpy as np
from osgeo import gdal, gdal_array
from tensorflow.keras.models import load_model

# 未学習画像のパス
input_dir = '/media/hina/AE60-1503/sabo_UNet_Planet_trial03/dataset_1/test/features'
output_dir = '/media/hina/AE60-1503/sabo_UNet_Planet_trial03/01_predict_masks/'
os.makedirs(output_dir, exist_ok=True)

# 学習済みモデルの読み込み
model = load_model('01_unet_model.h5')

# 画像読み込みと予測
for img_path in glob.glob(os.path.join(input_dir, '*.tif')):
    # 画像読み込み（Geo情報も保持）
    dataset = gdal.Open(img_path)
    image = dataset.ReadAsArray()
    image = np.moveaxis(image, 0, -1)  # (C, H, W) → (H, W, C)
    height, width, channels = image.shape

    # モデル入力に合わせて整形
    image_input = np.expand_dims(image, axis=0)  # (1, H, W, C)
    prediction = model.predict(image_input)[0]   # (H, W, 2)

    # クラス予測（argmaxで one-hot → class index）
    predicted_mask = np.argmax(prediction, axis=-1)  # (H, W)
    
    # 確率の取得（崩壊地の確率）
    collapse_prob = prediction[..., 1]  # (H, W) で、崩壊地の確率を取得

    # 閾値処理（確率が0.5以上で崩壊地、そうでない場合は非崩壊地）
    predicted_mask[collapse_prob < 0.59] = 0  # 崩壊地
    predicted_mask[collapse_prob >= 0.59] = 1  # 非崩壊地

    # 保存先パス
    filename = os.path.basename(img_path).replace('.tif', '_pred.tif')
    save_path = os.path.join(output_dir, filename)

    # GeoTIFFとして保存（元画像と同じジオリファレンス）
    driver = gdal.GetDriverByName('GTiff')
    out_raster = driver.Create(save_path, width, height, 1, gdal.GDT_Byte)
    out_raster.SetGeoTransform(dataset.GetGeoTransform())
    out_raster.SetProjection(dataset.GetProjection())
    out_raster.GetRasterBand(1).WriteArray(predicted_mask)
    out_raster.FlushCache()

    # 確率マップの保存（崩壊地の確率）
    prob_filename = os.path.basename(img_path).replace('.tif', '_prob.tif')
    prob_save_path = os.path.join(output_dir, prob_filename)

    out_prob_raster = driver.Create(prob_save_path, width, height, 1, gdal.GDT_Float32)
    out_prob_raster.SetGeoTransform(dataset.GetGeoTransform())
    out_prob_raster.SetProjection(dataset.GetProjection())
    out_prob_raster.GetRasterBand(1).WriteArray(collapse_prob)
    out_prob_raster.FlushCache()

    print(f"Saved: {save_path}")
    print(f"Probability map saved: {prob_save_path}")
