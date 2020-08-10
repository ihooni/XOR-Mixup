import tensorflow as tf
from tensorflow import keras
import numpy as np
import pandas as pd
from arguments import Args
import copy
from sklearn.manifold import MDS
# -*- coding: utf-8 -*-

import cv2
import matplotlib.pyplot as plt
from matplotlib import offsetbox
import os
import random

os.environ["CUDA_VISIBLE_DEVICES"] = '0'


def plot_data_with_images(data, images, ax=None, cmap='gray'):
    ax = ax or plt.gca()
    ax.scatter(data[:-1, 0], data[:-1, 1], c='k', s=100)
    ax.scatter(data[-1, 0], data[-1, 1], c='r', s=100)

    axis_lim = 2000
    ax.set_xlim(-axis_lim, axis_lim)
    ax.set_ylim(-axis_lim, axis_lim)

    for i in range(data.shape[0]):
        imagebox = offsetbox.AnnotationBbox(
            offsetbox.OffsetImage(images[i], zoom=1.2, cmap=cmap),
            data[i]
        )
        imagebox.patch.set_width(20)
        ax.add_artist(imagebox)


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

    args = Args().getParameters()

    is_test = False
    img_loc = "./train_images"

    # FL 위한 변수
    num_edge = 3
    num_max_image = int(5500 - (5500 % (num_edge + 1)))
    # Edge Server 가 가지고 있다고 가정할 Class 별 image의 index
    # 예 ) edge 0 : 0 ~ 200.png , edge 1 : 201 ~ 400.png ... , device 들은 여러개여도 한 array에
    idx_images = np.arange(num_max_image).reshape((num_edge + 1), int(num_max_image / (num_edge + 1)))

    # Data 관련 변수
    class_size = 10
    Target_class = 9
    average_num = 2
    skewed_data_num = 10
    normal_data_num = 200

    # 클래스 별 데이터 개수를 담고 있는 변수
    data_num = np.ones(shape=(10,)) * normal_data_num
    data_num[Target_class] = skewed_data_num

    # XOR 이미지를 만들때 Target Class 제외한 Class 들의 이미지를 균등하게 사용하기 위한 변수
    # dummy_num 만큼 각 class들은 XOR에 사용된다
    dummy_num = normal_data_num - data_num[Target_class]
    dummy_num = int(dummy_num / (class_size - 1))

    # 데이터 수 파악을 위한 변수들
    bef_num = copy.copy(data_num)
    af_num = copy.copy(data_num)

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
        images_history = []
        for idx_device in range(num_edge):

            # edge_(idx_device) 가 사용할 x, y 데이터 로드
            x_train = []  # Shape : (class_size * class별 이미지수 총합, )
            y_train = []  # Shape : (class_size * class별 이미지수 총합, )
            for i in range(class_size):
                temp = []
                for j in range(int(data_num[i])):
                    img = cv2.cvtColor(cv2.imread(
                        img_loc + '/User/' + str(int(i)) + '/' + str(int(idx_images[idx_device][j])) + '.png'),
                                       cv2.COLOR_BGR2GRAY)
                    x_train.append(img[:, :, np.newaxis].astype('float32') / 255)
                    y_train.append(keras.utils.to_categorical(i, class_size))
                    temp.append(img)

            # Device 가 가졌다고 가정하는 x 로드
            img_list = []  # Shape : (class_size , class별 이미지수)
            for i in range(class_size):
                temp = []
                for j in range(int(data_num[i])):
                    img = cv2.cvtColor(
                        cv2.imread(img_loc + '/User/' + str(int(i)) + '/' + str(int(idx_images[num_edge][j])) + '.png'),
                        cv2.COLOR_BGR2GRAY)
                    temp.append(img)
                img_list.append(temp)

            # Device_(idx_device) 의 model 생성
            models.append(model(args, idx_device))

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
                if i == Target_class:
                    continue

                # Dummy class 이미지와 target class 이미지의 XOR 수행
                for j in range(dummy_num):
                    # 같은 클래스의 이미지 average_num 개 만큼 average 수행한다.
                    # 이때 average 하는 이미지는 연속된 index 이미지 사용
                    # ex ) 0.png 1.png 2.png .... 최대 normal_data_num 까지

                    # Dummy Label Image
                    dummy_img = img_list[i][idx[j]]

                    # Target Label Image
                    start_idx = np.random.choice(np.arange(skewed_data_num), 1, replace=False)[0]
                    target_img = img_list[Target_class][idx2[int(start_idx)]]

                    train_label = np.zeros(shape=(class_size,))
                    train_label[Target_class] = 0.5
                    train_label[i] = 0.5

                    # MixUP
                    user_mix = cv2.addWeighted(target_img, 0.5, dummy_img, 0.5, 0)

                    images_history.append(np.concatenate((
                        np.expand_dims(dummy_img, axis=0),
                        np.expand_dims(target_img, axis=0),
                        np.expand_dims(user_mix, axis=0)
                    ), axis=0))

                    af_num[Target_class] += 1

                    train_img = user_mix[:, :, np.newaxis].astype('float32') / 255.
                    x_train.append(train_img)
                    y_train.append(train_label)

            x_train = np.array(x_train)
            y_train = np.array(y_train)

            models[idx_device].model.fit(x_train, y_train, epochs=epochs, batch_size=batch_size)
            models[idx_device].model.save('./save/linearMixUP/model_' + str(idx_device) + '.h5')

        # calculate MDS privacy and choose the minimum case
        mds = MDS(n_components=2, metric=True)
        min_privacy = 1e8
        min_case_mapped_images = None
        min_case_images = None
        for images in images_history:
            flatten = np.reshape(images, (images.shape[0], -1))
            mapped_images = mds.fit_transform(flatten)
            for i in range(len(mapped_images) - 1):
                privacy = np.linalg.norm(mapped_images[i] - mapped_images[-1])
                if privacy < min_privacy:
                    min_privacy = privacy
                    min_case_mapped_images = mapped_images
                    min_case_images = images

        # plot the MDS measurement results and save to image
        fig = plt.figure(figsize=(6, 6))
        ax = fig.add_subplot(1, 1, 1)
        plt.title(f'minimum privacy: {round(min_privacy)}', fontsize=15)
        plot_data_with_images(min_case_mapped_images, min_case_images, ax)
        plt.savefig(f'./save/linearMixUP/fig-{round(min_privacy, 3)}.png', dpi=300)


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
        # rate = np.zeros(shape=(num_edge,))
        # for i in range(num_edge):
        #    rate[i] = scores[i] / np.sum(scores)

        print("========================")
        print('Federated Learning Start')
        main_model = model(args, 4)
        for i in range(np.shape(models[0].model.layers)[0]):
            layer_w = np.array(models[0].model.layers[i].get_weights()) * 0.95
            for j in range(1, num_edge):
                layer_w += np.array(models[j].model.layers[i].get_weights()) * 0.025
            # layer_w = np.array(layer_w) / num_edge
            main_model.model.layers[i].set_weights(layer_w)

        main_model.model.save('./save/FL_linearMixUP/model_FL.h5')
        score = main_model.model.evaluate(x_test, y_test, batch_size=1000)
        print('[Main] Test loss:', score[0])
        print('[Main] Test accuracy:', score[1])

        predicted_result = main_model.model.predict(x_test)
        predicted_labels = np.argmax(predicted_result, axis=1)
        test_labels = np.argmax(y_test, axis=1)

        # 아래 부분의 내용 추가
        total_data = np.zeros(shape=(10,))
        wrong_result = np.zeros(shape=(10,))
        for n in range(0, len(test_labels)):
            total_data[test_labels[n]] += 1
            if predicted_labels[n] != test_labels[n]:
                wrong_result[test_labels[n]] += 1

        for i in range(10):
            acc = 1 - wrong_result[i] / total_data[i]
            print("Class [" + str(i) + "]" + " Acc : " + str(acc))

    else:
        # Test
        print("Test Procedure")
        main_model = model(args, 4)
        main_model.model = tf.keras.models.load_model('./save/linearMixUP/model_FL.h5')
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