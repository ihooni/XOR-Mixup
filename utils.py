import numpy as np
import random
import cv2
import matplotlib.pyplot as plt

class manager:
    def __init__(self, server_number, maximum = None, is_test = False):
        # TRAIN 데이터는 클래스 당 최대 5000 개
        # TEST 데이터는 클래스 당 최대 870 개
        # Server_number * DEFAULT_IMAGE_NUM
        # MNIST 각 클래스 내 폴더의 데이터 (ex) 0 클래스의 데이터는 0.png.....870.png까지 존재
        # 이 중 앞에 DEFAULT_IMGAE_NUM 개수는 Server에서 사용한다고 가정
        # 그러므로 해당 DEFAULT_IMAGE_NUM 이후의 데이터들을 유저가 가지고 있다고 가정할 것이다.
        DEFAULT_IMAGE_NUM = 30
        if is_test == True:
            if maximum == None:
                DATA_NUM = 870
            else:
                DATA_NUM = maximum
        else:
            if maximum == None:
                DATA_NUM = 5000
            else:
                DATA_NUM = maximum
        self.id_list = np.arange(server_number * DEFAULT_IMAGE_NUM, DATA_NUM)
        np.random.shuffle(self.id_list)

    def divide(self, user_number):
        temp = self.id_list[:np.shape(self.id_list)[0]-np.shape(self.id_list)[0]%user_number]
        return np.hsplit(temp, user_number)

class user:
    def __init__(self, name, list, is_test=False):
        self.name = str(name)
        self.id_list = list
        self.XOR_data = None
        # shape : num_class x 100
        self.XOR_id_list = None

        if is_test == True:
            img_loc = "test_images"
        else:
            img_loc = "train_images"

    def make_XOR(self, num, class_list, is_test=False):

        if is_test == True:
            img_loc = "./test_images"
        else:
            img_loc = "./train_images"

        # num : 1 클래스당 몇개의 데이터로 XOR을 만들 것인지 정한다
        # class_list : XOR로 만들기 원하는 class의 숫자가 들어있다
        # Step 1 : randomly select ids of each class
        np.random.shuffle(self.id_list)
        selected_ids = np.random.choice(self.id_list, num, replace=False)
        # Step 2 : make image
        # class 수만큼 for문 반복
        # 귀찮아서 대충 초기화 ...
        bit_xor = cv2.cvtColor(cv2.imread(img_loc + '/User/0/' + str(selected_ids[0]) + '.png'), cv2.COLOR_BGR2GRAY)
        bit_xor = cv2.bitwise_xor(bit_xor, bit_xor)

        # XOR
        for i in range(np.shape(class_list)[0]):
            for j in range(num):
                img = cv2.cvtColor(cv2.imread(img_loc + '/User/' + str(class_list[i]) + '/' + str(selected_ids[j]) + '.png'), cv2.COLOR_BGR2GRAY)
                bit_xor = cv2.bitwise_xor(bit_xor, img)
                # plt.imshow(img, cmap='gray')
                # plt.show()
        XOR_image = bit_xor
        self.XOR_id_list = selected_ids

        return selected_ids, XOR_image

class server:
    def __init__(self, server_number, num):
        # id_list : MNIST 데이터 셋에서 EGDE가 가지고 있다고 가정하는 image 수 DEFAULT_IMAGE_NUM x Server 수 
        # selected_ids : 각 서버가 가지고 있다고 가정하는 image를 랜덤하게 섞어 image의 숫자 (ex) 1.png , 3.png 와 같이 [1, 3, ...] 로 이름을 섞음
        # default_list : [server_number, 10 (MNIST의 CLASS 개수), 28, 28 (이미지 크기)] 각 서버별로 가지고 있는 defualt XOR Image
        DEFAULT_IMAGE_NUM = 10
        self.id_list = np.arange(0, DEFAULT_IMAGE_NUM * server_number)
        np.random.shuffle(self.id_list)
        self.selected_ids = self.divide(server_number)
        self.default_list = self.getDefault(server_number, num)

    def divide(self, server_number):
        return np.hsplit(self.id_list, server_number)

    def getDefault(self, server_number, num, is_test = False):

        if is_test == True:
            img_loc = "test_images"
        else:
            img_loc = "train_images"

        # Edge Server의 갯수
        # num : 각 클래스당 몇개의 이미지로 default XOR image를 가지는 것인지 나타냄
        images = []
        # XOR
        for i in range(server_number):
            temp = []
            for j in range(10):
                bit_xor = cv2.cvtColor(cv2.imread(img_loc + '/Server/0/0.png'), cv2.COLOR_BGR2GRAY)
                bit_xor = cv2.bitwise_xor(bit_xor, bit_xor)
                # class 내 default image 숫자 
                # k는 class 0 데이터 몇개씩 중첩시킬 것인지를 정함
                # k in range(1)이면 1개의 이미지만을 해당 클래스 대표 이미지로 설정
                for k in range(num):
                    img = cv2.cvtColor(cv2.imread(img_loc + '/Server/' + str(j) + '/' + str(self.selected_ids[i][k]) + '.png'), cv2.COLOR_BGR2GRAY)
                    bit_xor = cv2.bitwise_xor(bit_xor, img)
                temp.append(bit_xor)
            images.append(temp)
        return images

if __name__=="__main__":
    svr = server(3, 1)
    svr.getDefault(3, 1)
