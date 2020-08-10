import tensorflow as tf
from tensorflow import keras
import numpy as np
import pandas as pd
from arguments import Args
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

    is_test = True
    # ===================== #
    # Data 수 조절
    Target_class = [0, 2, 6, 8]
    skewd_value = 2
    normal_data_num = 200
    average_num = 2
    class_size = 10
    data_num = np.ones(shape=(10,))
    data_num *= normal_data_num

    for i in range(4):
        data_num[Target_class[i]] = skewd_value

    bef_num = np.ones(shape=(10,)) * normal_data_num
    for i in range(4):
        bef_num[Target_class[i]] = skewd_value

    af_num = np.ones(shape=(10,)) * normal_data_num
    for i in range(4):
        af_num[Target_class[i]] = skewd_value

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
        required_num = normal_data_num - data_num[Target_class[0]] * 4
        each = int(required_num / (class_size - 1))
        xor_data = []

        for p in range(4):
            for i in range(class_size):
                idx = np.random.choice(np.arange(normal_data_num), each, replace=False)
                idx2 = np.random.choice(np.arange(skewd_value), skewd_value, replace=False)

                if i == Target_class[0] or i == Target_class[1] or i == Target_class[2] or i == Target_class[3]:
                    continue

                for j in range(each):
                    tt = np.random.choice(np.arange(normal_data_num), average_num, replace=False)
                    img = img_list[i][idx[j]]
                    for k in range(1, average_num):
                        img2 = img_list[i][tt[k]]
                        img = cv2.addWeighted(img, 0.5, img2, 0.5, 0)

                    # User 의 Average
                    temp = np.random.choice(np.arange(skewd_value), average_num, replace=False)
                    user_avg = img_list[Target_class[p]][idx2[temp[0]]]
                    for k in range(1, average_num):
                        img_usr2 = img_list[Target_class[p]][idx2[temp[k]]]
                        user_avg = cv2.addWeighted(user_avg, 0.5, img_usr2, 0.5, 0)

                    # MDS 로 비교할 대상
                    # user_avg : 최종 User가 전송하는 이미지

                    # Server 의 XOR
                    temp_server = np.random.choice(np.arange(skewd_value, 500), average_num, replace=False)
                    ser_avg = cv2.cvtColor(cv2.imread(img_loc + '/Server/' + str(int(i)) + '/' + str(int(500)) + '.png'),
                                       cv2.COLOR_BGR2GRAY)
                    for k in range(1, average_num):
                        img_ser2 = cv2.cvtColor(cv2.imread(img_loc + '/Server/' + str(int(i)) + '/' + str(int(500 + int(k))) + '.png'),
                                        cv2.COLOR_BGR2GRAY)
                        ser_avg = cv2.addWeighted(ser_avg, 0.5, img_ser2, 0.5, 0)

                    # Server 의 재 XOR
                    bit_xor1 = cv2.bitwise_xor(img, user_avg)
                    bit_xor2 = cv2.bitwise_xor(bit_xor1, ser_avg)

                    # plt.subplot(511)
                    # plt.imshow(img, cmap='gray')
                    # plt.subplot(512)
                    # plt.imshow(user_avg, cmap='gray')
                    # plt.subplot(513)
                    # plt.imshow(ser_avg, cmap='gray')
                    # plt.subplot(514)
                    # plt.imshow(bit_xor1, cmap='gray')
                    # plt.subplot(515)
                    # plt.imshow(bit_xor2, cmap='gray')
                    # plt.show()

                    af_num[Target_class[p]] += 1

                    train_img = bit_xor2[:, :, np.newaxis].astype('float32') / 255.
                    x_train.append(train_img)
                    y_train.append(keras.utils.to_categorical(Target_class[p], class_size))

        x_train = np.array(x_train)
        y_train = np.array(y_train)

        print(np.shape(x_train))
        mod.model.fit(x_train, y_train, epochs=100, batch_size=256)
        mod.model.save('./save/normal_xor/model.h5')

        print(bef_num, af_num)
    else:
        # Test
        print("Test Procedure")
        mod.model = tf.keras.models.load_model('./save/normal_xor/model.h5')
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