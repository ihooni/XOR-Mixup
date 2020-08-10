import tensorflow as tf
from tensorflow import keras
import numpy as np
import pandas as pd
from arguments import Args
from sklearn.manifold import MDS
from sklearn.metrics import euclidean_distances
# -*- coding: utf-8 -*-

import cv2
import matplotlib.pyplot as plt
from utils import manager, user, server
import os
import random
os.environ["CUDA_VISIBLE_DEVICES"] = '0'

class model(object):
    def __init__(self, args, num):
        self.num = num
        self.model = self.build()

    def build(self):
        input_layer = keras.layers.Input(shape=(28, 28, 1), name='image_input')
        conv1 = keras.layers.Conv2D(56, (3, 3), activation='relu')(input_layer)
        pool1 = keras.layers.MaxPooling2D(2, 2)(conv1)
        conv2 = keras.layers.Conv2D(56, (3, 3), activation='relu')(pool1)
        pool2 = keras.layers.MaxPooling2D(2, 2)(conv2)
        flatten = keras.layers.Flatten()(pool2)
        dense1 = keras.layers.Dense(784, activation='relu')(flatten)
        output_layer = keras.layers.Dense(10, activation='softmax')(dense1)
        model = keras.models.Model(inputs=input_layer, outputs=output_layer, name='model' + str(self.num))
        model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
        # model.summary()
        return model


if __name__ == "__main__":

    is_test = False
    # ===================== #
    # Data 수 조절
    Target_class = 9
    skewd_value = 2
    normal_data_num = 60
    average_num = 2
    class_size = 10
    data_num = np.ones(shape=(10,))
    data_num *= normal_data_num
    data_num[Target_class] = skewd_value

    bef_num = np.ones(shape=(10,)) * normal_data_num
    bef_num[Target_class] = skewd_value
    af_num = np.ones(shape=(10,)) * normal_data_num
    af_num[Target_class] = skewd_value

    args = Args().getParameters()

    (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

    input_shape = (28, 28, 1)
    x_train = x_train.reshape(x_train.shape[0], 28, 28, 1)
    x_test = x_test.reshape(x_test.shape[0], 28, 28, 1)

    x_train = x_train.astype('float32') / 255.
    x_test = x_test.astype('float32') / 255.

    print('x_train shape:', x_train.shape)
    print(x_train.shape[0], 'train samples')
    print(x_test.shape[0], 'test samples')

    batch_size = 128
    num_classes = 10
    epochs = 12

    y_train = keras.utils.to_categorical(y_train, num_classes)
    y_test = keras.utils.to_categorical(y_test, num_classes)

    mod = model(args, 0)

    if is_test == False:
        img_loc = "./train_images"
    else:
        img_loc = "./test_images"


    img_list = []
    x_train = []
    y_train = []
    for i in range(class_size):
        temp = []
        for j in range(int(data_num[i])):
            img = cv2.cvtColor(cv2.imread(img_loc + '/User/' + str(int(i)) + '/' + str(int(j)) + '.png'),
                               cv2.COLOR_BGR2GRAY)
            x_train.append(img[:, :, np.newaxis].astype('float32') / 255)
            y_train.append(keras.utils.to_categorical(i, class_size))
            temp.append(img)
        img_list.append(temp)

    if is_test == False:
        # ================================ #
        # XOR Data 만들기
        required_num = normal_data_num - data_num[Target_class]
        each = int(required_num / (class_size - 1))
        xor_data = []
        for i in range(class_size):
            idx = np.random.choice(np.arange(normal_data_num), each, replace=False)
            idx2 = np.random.choice(np.arange(skewd_value), skewd_value, replace=False)
            if i == Target_class:
                continue

            for j in range(each):
                img = img_list[i][idx[j]]
                # User 의 Linear MixedUp
                temp = np.random.choice(np.arange(skewd_value), skewd_value, replace=False)[0]
                target = img_list[Target_class][idx2[temp]]
                user_mix = cv2.addWeighted(img, 0.5, target, 0.5, 0)

                #MDS 로 비교할 대상
                # user_mix : 최종 User가 전송하는 이미지

                af_num[Target_class] += 1

                train_img = user_mix[:, :, np.newaxis].astype('float32') / 255.
                x_train.append(train_img)
                train_label = np.zeros(shape=(class_size,))
                train_label[Target_class] = 0.5
                train_label[i] = 0.5
                y_train.append(train_label)

        x_train = np.array(x_train)
        y_train = np.array(y_train)

        print(np.shape(x_train))
        mod.model.fit(x_train, y_train, epochs=100, batch_size=256)
        mod.model.save('./save/normal_linear/model.h5')

        print(bef_num, af_num)
    else:
        # Test
        print("Test Procedure")
        mod.model = tf.keras.models.load_model('./save/normal_linear/model.h5')
        score = mod.model.evaluate(x_test, y_test, batch_size=1000)
        print('Test loss:', score[0])
        print('Test accuracy:', score[1])

        predicted_result = mod.model.predict(x_test)
        predicted_labels = np.argmax(predicted_result, axis=1)
        test_labels = np.argmax(y_test, axis=1)

        total_data = np.zeros(shape=(10,))
        wrong_result = np.zeros(shape=(10,))
        for n in range(0, len(test_labels)):
            total_data[test_labels[n]] += 1
            if predicted_labels[n] != test_labels[n]:
                wrong_result[test_labels[n]] += 1

        for i in range(10):
            acc = 1 - wrong_result[i] / total_data[i]
            print("Class [" + str(i) + "]" + " Acc : " + str(acc))