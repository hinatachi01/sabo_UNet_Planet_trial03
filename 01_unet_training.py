import glob
import numpy as np
from osgeo import gdal
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping

# 必要なデータ量の目安
## Train画像数: 1,000～1,500枚
## Val画像数: 300～500枚

# データセットのパス
train_image_dir = '/media/hina/AE60-1503/sabo_UNet_Planet_trial03/dataset_1/train/features/'
train_mask_dir = '/media/hina/AE60-1503/sabo_UNet_Planet_trial03/dataset_1/train/masks/'

val_image_dir = '/media/hina/AE60-1503/sabo_UNet_Planet_trial03/dataset_1/val/features/'
val_mask_dir = '/media/hina/AE60-1503/sabo_UNet_Planet_trial03/dataset_1/val/masks/'

# データファイルをndarrayでリストに収納
train_images = []
train_masks = []

val_images = []
val_masks = []

for train_image_files in glob.glob(train_image_dir + "*.tif"):
    img = gdal.Open(train_image_files).ReadAsArray()
    img = np.moveaxis(img, 0, -1)
    train_images.append(img)

for train_mask_files in glob.glob(train_mask_dir + "*.tif"):
    msk = gdal.Open(train_mask_files).ReadAsArray()
    msk = np.nan_to_num(msk, nan=1)
    msk = to_categorical(msk, num_classes=2)
    train_masks.append(msk)

for val_image_files in glob.glob(val_image_dir + "*.tif"):
    img = gdal.Open(val_image_files).ReadAsArray()
    img = np.moveaxis(img, 0, -1)
    val_images.append(img)

for val_mask_files in glob.glob(val_mask_dir + "*.tif"):
    msk = gdal.Open(val_mask_files).ReadAsArray()
    msk = np.nan_to_num(msk, nan=1)
    msk = to_categorical(msk, num_classes=2)
    val_masks.append(msk)

train_images = np.array(train_images, dtype=np.float32)
train_masks = np.array(train_masks, dtype=np.float32)

val_images = np.array(val_images, dtype=np.float32)
val_masks = np.array(val_masks, dtype=np.float32)


from tensorflow.keras import layers, models, optimizers
from tensorflow.keras.optimizers import Adam

# モデル定義（softmax & 2クラス）
def unet_model(input_size=(128, 128, 4), num_classes=2):
    inputs = layers.Input(input_size)
    
    # 第一層
    x = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(inputs)
    x = layers.MaxPooling2D((2, 2))(x)
    
    # 第二層
    x = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(x)
    x = layers.MaxPooling2D((2, 2))(x)
    
    # 第三層
    x = layers.Conv2D(256, (3, 3), activation='relu', padding='same')(x)
    x = layers.MaxPooling2D((2, 2))(x)

    # アップサンプリング層
    x = layers.UpSampling2D((2, 2))(x)
    x = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(x)

    # アップサンプリング層
    x = layers.UpSampling2D((2, 2))(x)
    x = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(x)
    
    x = layers.UpSampling2D((2, 2))(x)
    outputs = layers.Conv2D(num_classes, (1, 1), activation='softmax')(x)

    model = models.Model(inputs, outputs)

    model.compile(
        optimizer=Adam(learning_rate=1e-5),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    return model

# モデルの作成
model = unet_model()

# 早期停止のコールバック設定
early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

# 崩壊地 (0): 重み2.0、非崩壊地 (1): 重み1.0
weight_map = np.array([2.0, 1.0])
train_sample_weights = np.sum(train_masks * weight_map, axis=-1)  # (N, H, W)

# 損失関数
loss='categorical_crossentropy'

# モデルの訓練
model.fit(
    train_images, train_masks, 
    sample_weight=train_sample_weights,
    epochs=10, 
    batch_size=16, 
    validation_data=(val_images, val_masks),
    verbose=1,
    callbacks=[early_stopping]
)

# モデルの保存
model.save('01_unet_model.h5')