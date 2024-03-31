from email.mime import application
import tensorflow as tf  
from keras.preprocessing.image import ImageDataGenerator  
from torch import ResNet50, preprocess_input, decode_predictions  
from keras.models import Model  
from keras.layers import Dense, GlobalAveragePooling2D  
from keras.optimizers import Adam  
from keras.callbacks import ModelCheckpoint, TensorBoard  
import matplotlib.pyplot as plt  
import numpy as np  
import os  
import keras as  kk
import pandas as pd  
from sklearn.model_selection import train_test_split
  
# 1. 数据集下载和解压  
 
data = pd.read_csv('D:\python\2023090910010-李骁涵-机器学习\test4\数据集.zip.baiduyun.p.downloading') 
X = data.drop('target', axis=1)  # 特征  
y = data['target']
train_dir,test_dir = train_test_split(X, y, test_size=0.8) 
  
# 2. 数据加载和预处理  
train_datagen = ImageDataGenerator(  
    rescale=1./255,  
    rotation_range=20,  
    width_shift_range=0.2,  
    height_shift_range=0.2,  
    shear_range=0.2,  
    zoom_range=0.2,  
    horizontal_flip=True,  
    fill_mode='nearest'  
)  
  
test_datagen = ImageDataGenerator(rescale=1./255)  
  
train_generator = train_datagen.flow_from_directory(  
    train_dir,  
    target_size=(32, 32),  
    batch_size=32,  
    class_mode='categorical'  
)  
  
test_generator = test_datagen.flow_from_directory(  
    test_dir,  
    target_size=(32, 32),  
    batch_size=32,  
    class_mode='categorical'  
)  
  
# 3. 定义ResNet模型  
base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(32, 32, 3))  
  
# 添加全局平均池化层  
x = base_model.output  
x = GlobalAveragePooling2D()(x)  
  
# 添加全连接层  
x = Dense(1024, activation='relu')(x)  
predictions = Dense(train_generator.num_classes, activation='softmax')(x)  
  
# 构建最终的模型  
model = Model(inputs=base_model.input, outputs=predictions)  
  
# 只训练顶层权重  
for layer in base_model.layers:  
    layer.trainable = False  
  
# 编译模型 (使用特定的损失函数编译模型) 
model.compile(optimizer=Adam(lr=0.0001), loss='categorical_crossentropy', metrics=['accuracy'])  
  
# 4. 模型训练 (用编译好的模型进行训练)
checkpointer = ModelCheckpoint(filepath='model.h5', verbose=1, save_best_only=True)  
tensorboard = TensorBoard(log_dir='./logs')  
  
history = model.fit(  
    train_generator,  
    steps_per_epoch=train_generator.samples // train_generator.batch_size,  
    epochs=10,  
    validation_data=test_generator,  
    validation_steps=test_generator.samples // test_generator.batch_size,  
    callbacks=[checkpointer, tensorboard]  
)  
  
# 5. 模型评估  
model.load_weights('model.h5')  # 加载最佳权重  
test_loss, test_acc = model.evaluate(test_generator)  
print(f'Test accuracy: {test_acc}')  
  
# 6. 可视化训练过程  
plt.plot(history.history['accuracy'], label='train accuracy')  
plt.plot(history.history['val_accuracy'], label='test accuracy')  
plt.title('Accuracy')  
plt.ylabel('accuracy')  
plt.xlabel('epoch')  
plt.legend(loc='upper left')  
plt.show()  
  
plt.plot(history.history['loss'], label='train loss')  
plt.plot(history.history['val_loss'], label='test loss')  
plt.title('Loss')