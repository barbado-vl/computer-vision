import os
import pandas as pd
import tensorflow as tf
import tensorflow.keras.backend as K
from sklearn.model_selection import train_test_split
import cv2
import matplotlib.pyplot as plt
import multiprocessing
import time
import numpy as np
import gc
from albumentations import HorizontalFlip, RandomBrightnessContrast, Compose, Normalize, Resize

# глянуть сколько файлов_________________________________________________________________________________
#print("Train set: ", len(os.listdir("\ML_carvana/train")))
#print("Train masks: ", len(os.listdir("\ML_carvana/train_masks")))

# _______________________________________________________________________________________________________
# нужно организовать файлы изображений в структуру данных 
# нужен файл и его id, потом собрать их в списки, которые поместить в специальную структуру (массив из списков),
# полученна структура будет использоваться далее
car_ids = []
paths = []
for dirname, _, filenames in os.walk('\ML_carvana/train'):  # walk возвращает 3 переменных, dirpath не нужна, поэтому "_"
    for filename in filenames:
        path = os.path.join(dirname, filename)    # путь до файла
        paths.append(path)

        car_id = filename.split(".")[0]       # как id берем имя файла без его расширения 
        car_ids.append(car_id)

d = {"id": car_ids, "car_path": paths}   # словарь, конкретнее, массив из 2-х колонок
df = pd.DataFrame(data = d)
df = df.set_index('id')              # для объекта df надо индекс указать - это будет id колонка

# тоже самое для файлов масок
car_ids = []
mask_paths = []
for dirname, _, filenames in os.walk('\ML_carvana/train_masks'): 
    for filename in filenames:
        path = os.path.join(dirname, filename)   
        mask_paths.append(path)

        car_id = filename.split(".")[0]       
        car_id = car_id.split("_mask")[0]    # _mask убираем
        car_ids.append(car_id)

d = {"id": car_ids, "mask_path": mask_paths}  
mask_df = pd.DataFrame(data = d)
mask_df = mask_df.set_index('id')

# в "табличку df еще добавляем колонку путей до mask файлов
df["mask_path"] = mask_df["mask_path"]
# глянуть что в dtataframe
#print(df.head(5))      # смотрим какие данные по первым 5 строчкам таблицы dataframe________________

# обработка самих изображений_______________________________________________________________________________
img_size = [224, 224]

# поворачиваем изображение и меняем яркость у файлом, которым random больше 0,5 выпадет
# это даст больше вариантов для тренировки
def data_augmentation(car_img, mask_img):
    if tf.random.uniform(()) > 0.5:
        car_img = tf.image.flip_left_right(car_img)
        mask_img = tf.image.flip_left_right(mask_img)
    if tf.random.uniform(()) > 0.5:
        car_img = tf.image.random_brightness(car_img, max_delta=0.05, seed=42)
    return car_img, mask_img

# обработка изображений
def preprocessing(car_path, mask_path):
    car_img = tf.io.read_file(car_path)                  # получаем файл по пути
    car_img = tf.image.decode_jpeg(car_img, channels=3)  # декодируем в тензор(поле, вектор, набор чисел) 3 значит RGB стандарт 
    car_img = tf.image.resize(car_img, img_size)         # меняем размер

    car_img = tf.cast(car_img, tf.float32) / 255.0   # переводим значение пикселя в float формат и делим на 255 -- зачем??
                                                     
    mask_img = tf.io.read_file(mask_path)                 # тоже самое как для car_img
    mask_img = tf.image.decode_jpeg(mask_img, channels=3)
    mask_img = tf.image.resize(mask_img, img_size)

    mask_img = mask_img[:,:,:1]                          # превращаем в one hot encoding vector??

    mask_img = tf.math.sign(mask_img)

    return car_img, mask_img

# DATASET создаем для тренировки и проверки_____________________________________________________________________
# сперва сама функция
def create_dataset(df, train = False):
    ds = tf.data.Dataset.from_tensor_slices((df["car_path"].values, df["mask_path"].values))  # передаем в функцию tf картеж из списков
    ds = ds.map(preprocessing, tf.data.AUTOTUNE)  # через map применяем функцию обработку preprocessing
    if train:
        ds = ds.map(data_augmentation, tf.data.AUTOTUNE)  # через map применяем функцию data_augmentation
    return ds

# создаем сами наборы данных
train_df, valid_df = train_test_split(df, random_state=42, test_size=.25) # разбиваем на наборы с помощью функции из sklearn
train = create_dataset(train_df, train = True)  # тренировачный набор
valid = create_dataset(valid_df, train = False)  # валидационный

#print (train_df.shape, valid_df.shape)    # смотрим как поделило____

# константы
TRAIN_LENGTH = len(train_df)
BATCH_SIZE = 16
BUFFER_SIZE = 1000

# обработка дата сетов
train_dataset = train.cache()
train_dataset = train.shuffle(BUFFER_SIZE).batch(BATCH_SIZE).repeat() # кэшируем, смешиваем, делим на батчи?
train_dataset = train_dataset.prefetch(buffer_size=tf.data.AUTOTUNE)  # какие-то prefetch ставим... границы итераций??
valid_dataset = valid.batch(BATCH_SIZE)

# НЕЙРОННАЯ СЕТЬ________________________________________________________________________________ 
# создаем encoder
base_model = tf.keras.applications.MobileNetV2(
    input_shape=[224, 224, 3], # 3 канала RGB кодировки пикселя,
    include_top=False          # это отключаем последний слом, т.к.нужны имбрединги, а не классификация (что?)
)                     
layer_names = [
    'block_1_expand_relu',   # 64x64
    'block_3_expand_relu',   # 32x32
    'block_6_expand_relu',   # 16x16
    'block_13_expand_relu',  # 8x8
    'block_16_project'       # 4x4
    ]
base_model_outputs = [base_model.get_layer(name).output for name in layer_names] # список слоев создаем
down_stack = tf.keras.Model(inputs=base_model.input, outputs=base_model_outputs) # из списка модель
down_stack.trainable = False # т.к. тренировать не надо, только имбрединг(?), то False

# создаем decoder
def upsample(filters, size):
    initializer = tf.random_normal_initializer(0., 0.02)
    result = tf.keras.Sequential()
    result.add(
        tf.keras.layers.Conv2DTranspose(
            filters,
            size,
            strides=2,
            padding='same',
            kernel_initializer=initializer,
            use_bias=False
        )
    )
    result.add(tf.keras.layers.BatchNormalization())
    return result

up_stack = [
    upsample(512, 3), # 4x4 -> 8x8
    upsample(256, 3), # 8x8 -> 16x16
    upsample(128, 3), # 16x16 -> 32x32
    upsample(64, 3)   # 32x32 -> 64x64
]

# реализация Unet где вызываем encoder и decoder, созданные выше
def unet_model():
    inputs = tf.keras.layers.Input(shape = [224, 224, 3])
    skips = down_stack(inputs)  # вызываем encoder

    x = skips[-1]
    skips = reversed(skips[:-1])

    for up, skip in zip(up_stack, skips):  # вызываем decoder
        x = up(x)
        concat = tf.keras.layers.Concatenate()  # конкатенируем слои из обоих, по Unet модели сети
        x = concat([x, skip])
        
    last = tf.keras.layers.Conv2DTranspose(  # используем последний слой для классификации пикселей
        1,
        3,
        strides=2,
        activation='sigmoid',
        padding='same'
    )
    x = last(x)
    return tf.keras.Model(inputs=inputs, outputs=x)

# Расчет коэффициента Дайса__________________________________________________________________________
def dice_coef(y_true, y_pred, smooth = 1):
    intersection = K.sum(y_true * y_pred, axis=[1, 2, 3])
    union = K.sum(y_true, axis=[1, 2, 3]) + K.sum(y_pred, axis=[1, 2, 3])
    return K.mean((2. * intersection + smooth) / (union + smooth), axis=0)

def dice_loss(y_true, y_pred):
    return 1 - dice_coef(y_true, y_pred)


# Визуализация______________________________________________________________________________________
#sample = cv2.imread('F:\ML_carvana/train/0cdf5b5d0ce1_01.jpg')
#sample = cv2.cvtColor(sample, cv2.COLOR_BGR2RGB)

def display(display_list):
    #plt.imshow(data)
    #plt.show(block=False)
    #plt.show()____
    plt.figure(figsize=(15, 15))
    title = ['Input Image', 'True Mask', 'Predicted Mask']
    for i in range(len(display_list)):
        plt.subplot(1, len(display_list), i+1)
        plt.title(title[i] + f' of shape {display_list[i].shape}')
        plt.imshow(tf.keras.preprocessing.image.array_to_img(display_list[i]))
        plt.axis('off')
    plt.show()

for images, mask in train_dataset.take(1):
    sample_image = images[0]
    sample_mask = mask[0]

def show_predictions(sample_image, sample_mask):
    pred_mask = model.predict(sample_image[tf.newaxis, ...])
    pred_mask = pred_mask.reshape(img_size[0], img_size[1], 1)
    display([sample_image, sample_mask, pred_mask])


# ЗАПУСК МОДЕЛИ____________________________________________________________________________________
model = unet_model()

# Запуск уже с обучением
class DisplayCallback(tf.keras.callbacks.Callback):      # представление результата по каждой 3 эпохе (условие if)
    def on_epoch_begin(self, epoch, logs=None):
        if (epoch + 1) % 3 == 0:
            multiprocessing.Process(target=show_predictions(sample_image, sample_mask), args=([1, 2, 3],)).start()

early_stop = tf.keras.callbacks.EarlyStopping(patience=4, restore_best_weights=True) 
                                                    # контроль работы модели ... если при 4 эпохах ... вернет лучшие...

sv_mod = tf.keras.callbacks.ModelCheckpoint(        # функция для сохранение модели, пойдет параметром в модель
    'mobnet_v2_best.hdf5',
    monitor='val_loss',
    save_best_only=True,
    period=1
)


EPOCHS = 10
STEPS_PER_EPOCH = TRAIN_LENGTH // BATCH_SIZE

if __name__ == "__main__":
    print("starting __main__")
    
    #Тестовый запуск без обучения
    #model.compile(
    #   optimizer='adam',
    #   loss = dice_loss,
    #   metrics = [dice_coef, 'binary_accuracy']
    #)
    #multiprocessing.Process(target=show_predictions(sample_image, sample_mask), args=([1, 2, 3],)).start()
    
    #Запуск с обучением
    model_history = model.fit(
        train_dataset,
        epochs=EPOCHS,
        steps_per_epoch=STEPS_PER_EPOCH,
        validation_data=valid_dataset,
        callbacks=[DisplayCallback(), early_stop, sv_mod]
    )

    time.sleep(5)
    print("exiting main")
    os._exit(0) # this exits immediately with no cleanup or buffer flushing