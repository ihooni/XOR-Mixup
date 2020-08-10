import tensorflow as tf
from tensorflow import keras
import numpy as np
import pandas as pd
# from arguments import Args
import copy
# -*- coding: utf-8 -*-

import cv2
import matplotlib.pyplot as plt
# from utils import manager, user, server
import os
import random
os.environ["CUDA_VISIBLE_DEVICES"] = '0'

class model(object):
    def __init__(self, num):
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

    # args = Args().getParameters()

    is_test = False
    img_loc = "./train_images"

    # FL 위한 변수
    num_edge = 3
    num_max_image = int(5500 - (5500 % (num_edge + 1)))
    # Edge Server 가 가지고 있다고 가정할 Class 별 image의 index
    # 예 ) edge 0 : 0 ~ 200.png , edge 1 : 201 ~ 400.png ... , device 들은 여러개여도 한 array에
    idx_images = np.arange(num_max_image).reshape((num_edge + 1), int(num_max_image/(num_edge + 1)))

    # Data 관련 변수
    class_size = 10
    # Target_class = 5
    Target_classes = [0]
    average_num = 2
    skewed_data_num = 10    # n
    normal_data_num = 200   # m

    # 클래스 별 데이터 개수를 담고 있는 변수
    data_num = np.ones(shape=(10,)) * normal_data_num
    # data_num[Target_class] = skewed_data_num
    for target_class in Target_classes:
        data_num[target_class] = skewed_data_num

    # XOR 이미지를 만들때 Target Class 제외한 Class 들의 이미지를 균등하게 사용하기 위한 변수
    # dummy_num 만큼 각 class들은 XOR에 사용된다
    dummy_num = normal_data_num - len(Target_classes) * skewed_data_num
    dummy_num = int(dummy_num / (class_size - len(Target_classes)))

    # 데이터 수 파악을 위한 변수들
    bef_num = copy.copy(data_num)
    af_num  = copy.copy(data_num)

    ##################
    # MNIST 다운로드 #
    (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
    input_shape = (28, 28, 1)
    x_train = x_train.reshape(x_train.shape[0], 28, 28, 1)
    x_test = x_test.reshape(x_test.shape[0], 28, 28, 1)
    x_train = x_train.astype('float32') / 255.
    x_test = x_test.astype('float32') / 255.
    ##################

    ############################
    # Label의 one-hot encoding #
    y_train = keras.utils.to_categorical(y_train, class_size)
    y_test = keras.utils.to_categorical(y_test, class_size)

    batch_size = 128
    num_classes = 10
    epochs = 100

    if is_test == False:

        # edge 개수만큼 학습을 반복
        # 학습된 edge 들의 weight를 모아서 평균낸 모델을 만들어서
        # 최종 모델의 Accuracy 측정이 FL 실험
        models = []
        for idx_device in range(num_edge):

            # edge_(idx_device) 가 사용할 x, y 데이터 로드
            x_train = []  # Shape : (class_size * class별 이미지수 총합, )
            y_train = []  # Shape : (class_size * class별 이미지수 총합, )
            for i in range(class_size):
                temp = []
                for j in range(int(data_num[i])):
                    img = cv2.cvtColor(cv2.imread(img_loc + '/User/' + str(int(i)) + '/' + str(int(idx_images[idx_device][j])) + '.png'),
                                       cv2.COLOR_BGR2GRAY)
                    x_train.append(img[:, :, np.newaxis].astype('float32') / 255)
                    y_train.append(keras.utils.to_categorical(i, class_size))
                    temp.append(img)

            # Device 가 가졌다고 가정하는 x 로드
            img_list = [] # Shape : (class_size , class별 이미지수)
            for i in range(class_size):
                temp = []
                for j in range(int(data_num[i])):
                    img = cv2.cvtColor(cv2.imread(img_loc + '/User/' + str(int(i)) + '/' + str(int(idx_images[num_edge][j])) + '.png'),
                                       cv2.COLOR_BGR2GRAY)
                    temp.append(img)
                img_list.append(temp)

            # Device_(idx_device) 의 model 생성
            models.append(model(idx_device))

            for target_class in Target_classes:
                # Device 가 Edge Server로 보낼 XOR 데이터를 생성
                # idx_images 변수의 0~num_edge 까지는 edge가 가진 이미지 index로 가정
                # num_edge+1 ~ 끝까지는 device가 가진 이미지 index로 가정
                xor_data = []
                for i in range(class_size):
                    # idx : device 가 가진 이미지의 index 들을 랜덤으로 suffle
                    # idx2 : target label 의 이미지는 skewed_data_num 으로 가정된 숫자만큼 있다고 가정하므로, 해당 index 내에서만 뽑음
                    idx = np.random.choice(np.arange(normal_data_num), dummy_num, replace=False)
                    idx2 = np.random.choice(np.arange(skewed_data_num), skewed_data_num, replace=False)

                    # Target class 내의 이미지 끼리 XOR 하는 것을 방지
                    if i in Target_classes:
                        continue

                    # Dummy class 이미지와 target class 이미지의 XOR 수행
                    for j in range(dummy_num):
                        # 같은 클래스의 이미지 average_num 개 만큼 average 수행한다.
                        # 이때 average 하는 이미지는 연속된 index 이미지 사용
                        # ex ) 0.png 1.png 2.png .... 최대 normal_data_num 까지

                        # Dummy Label Image의 Average
                        img = img_list[i][idx[j]]
                        for k in range(1, average_num):
                            img2 = img_list[i][(idx[j] + k) % normal_data_num]
                            img = cv2.addWeighted(img, 0.5, img2, 0.5, 0)

                        # Target Label Image의 Average
                        start_idx = np.random.choice(np.arange(skewed_data_num), 1, replace=False)[0]
                        user_avg = img_list[target_class][idx2[int(start_idx)]]
                        for k in range(1, average_num):
                            img_usr2 = img_list[target_class][idx2[(start_idx + k) % skewed_data_num]]
                            user_avg = cv2.addWeighted(user_avg, 0.5, img_usr2, 0.5, 0)

                        # Edge 가 reXOR을 위한 dummy label 이미지의 average image를 만듬
                        # idx_images[idx_device] 에 Edge가 가진 image index가 들어있음
                        idx_edge = np.random.choice(copy.copy(idx_images[idx_device]), 1, replace=False)
                        ser_avg = cv2.cvtColor(
                            cv2.imread(img_loc + '/Server/' + str(int(i)) + '/' + str(int(idx_edge[0])) + '.png'),
                            cv2.COLOR_BGR2GRAY)
                        for k in range(1, average_num):
                            img_ser2 = cv2.cvtColor(cv2.imread(
                                img_loc + '/Server/' + str(int(i)) + '/' + str(int((idx_edge[0] + k) % 500)) + '.png'),
                                                    cv2.COLOR_BGR2GRAY)
                            ser_avg = cv2.addWeighted(ser_avg, 0.5, img_ser2, 0.5, 0)

                        # Edge 의 재 XOR
                        # bit_xor1 : Device 가 Edge로 보냈다고 가정하는 XOR 이미지
                        # bit_xor2 : Edge가 학습을 위해서 사용할 reXOR 이미지
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

                        af_num[target_class] += 1

                        train_img = bit_xor2[:, :, np.newaxis].astype('float32') / 255.
                        x_train.append(train_img)
                        y_train.append(keras.utils.to_categorical(target_class, class_size))

            x_train = np.array(x_train)
            y_train = np.array(y_train)

            models[idx_device].model.fit(x_train, y_train, epochs=epochs, batch_size=batch_size)
            models[idx_device].model.save('./save/FL/model_' + str(idx_device) + '.h5')

        ####################################################################
        # Edge들 학습이 끝나면 학습된 3개의 모델의 Weight 들을 모아서 평균냄
        scores = np.zeros(shape=(num_edge,))
        score = models[0].model.evaluate(x_test, y_test, batch_size=1000)
        print('[Worker 1] Test loss:', score[0])
        print('[Worker 1] Test accuracy:', score[1])
        scores[0] = score[1]
        score = models[1].model.evaluate(x_test, y_test, batch_size=1000)
        print('[Worker 2] Test loss:', score[0])
        print('[Worker 2] Test accuracy:', score[1])
        scores[1] = score[1]
        score = models[2].model.evaluate(x_test, y_test, batch_size=1000)
        print('[Worker 3] Test loss:', score[0])
        print('[Worker 3] Test accuracy:', score[1])
        scores[2] = score[1]

        # 비율 계산
        #rate = np.zeros(shape=(num_edge,))
        #for i in range(num_edge):
        #    rate[i] = scores[i] / np.sum(scores)

        print("========================")
        print('Federated Learning Start')
        main_model = model(4)
        for i in range(np.shape(models[0].model.layers)[0]):
            layer_w = np.array(models[0].model.layers[i].get_weights()) * 0.95
            for j in range(1, num_edge):
                layer_w += np.array(models[j].model.layers[i].get_weights()) * 0.025
            #layer_w = np.array(layer_w) / num_edge
            main_model.model.layers[i].set_weights(layer_w)

        main_model.model.save('./save/FL/model_FL.h5')
        score = main_model.model.evaluate(x_test, y_test, batch_size=1000)
        print('[Main] Test loss:', score[0])
        print('[Main] Test accuracy:', score[1])

        # predicted_result = main_model.model.predict(x_test)
        # predicted_labels = np.argmax(predicted_result, axis=1)
        # test_labels = np.argmax(y_test, axis=1)
        #
        # # 아래 부분의 내용 추가
        # total_data = np.zeros(shape=(10,))
        # wrong_result = np.zeros(shape=(10,))
        # for n in range(0, len(test_labels)):
        #     total_data[test_labels[n]] += 1
        #     if predicted_labels[n] != test_labels[n]:
        #         wrong_result[test_labels[n]] += 1
        #
        # for i in range(10):
        #     acc = 1 - wrong_result[i] / total_data[i]
        #     print("Class [" + str(i) + "]" + " Acc : " + str(acc))

    else:
        # Test
        print("Test Procedure")
        main_model = model(4)
        main_model.model = tf.keras.models.load_model('./save/FL/model_FL.h5')
        score = main_model.model.evaluate(x_test, y_test, batch_size=1000)
        print('Test loss:', score[0])
        print('Test accuracy:', score[1])

        predicted_result = main_model.model.predict(x_test)
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