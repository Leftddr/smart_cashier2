import tensorflow as tf
from tensorflow.compat.v1 import keras
from tensorflow.compat.v1.keras import utils
from tensorflow.compat.v1.keras import layers
from tensorflow.compat.v1.keras import datasets
from tensorflow.compat.v1.keras.callbacks import EarlyStopping
from tensorflow.compat.v1.keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D
import numpy as np
import matplotlib.pyplot as plt
import PIL.Image as pil
import os.path
import time
import cv2
from tensorflow.compat.v1.keras import backend as K

#데이터 폴더의 이미지를 LOADING 한다.
data_folder = "./test_image/"
MINIMUM_IMAGE_NUM = 501

#훈련중인 모델을 저장할 경로와 파일 이름
checkpoint_path = 'model'
checkpoint_dir = './checkpoints/'

train_images = []
train_labels = []
test_images = []
test_labels = []

#우리가 분류해야될 물품의 목록을 모아놓는다.
class_names = [
    'blackbean', 'herbsalt', 'homerun', 'lion', 'narangd', 'rice', 'sixopening', 'skippy', 'BlackCap', 'CanBeer', 'doritos',
    'Glasses', 'lighter', 'mountaindew', 'pepsi', 'Spoon',  'tobacco', 'WhiteCap'
]

test_names = [
    'testsetA',
]
dict_class_names = None

batch_size = 1000
num_classes = len(class_names)
epochs = 35
model = []
#최종 accuracy가 가장 높은 model을 저장해 놓는다.
final_model = None
#각 훈련 집합마다 accuracy를 측정하여 가장 높은 정확도를 가지는 모델을 가지고 있는다.
global_accuracy = 0.0

#class_names의 list를 0, 1, 2.... 숫자 label로 바꾼다
def list_to_dictionary():
    global dict_class_names
    dict_class_names = {class_names[i] : i for i in range(0, len(class_names))}

#우리가 모아놓은 폴더에서 image와 label를 load한다.
#데이터에 맞게 label를 load해야 한다. ex) 의자 : 0, 과자 : 1, 핸드폰 : 2....
def load_data():
    global train_images
    global train_labels
    global test_images
    global test_labels

    #label를 위한 키 값 정리
    list_to_dictionary() 
    #훈련 데이터 평균 size 하기
    #훈련 데이터 및 label 정의하기
    
    (width, height) = data_img_size()
    train_data_img(width, height)   
    test_data_img(width, height)
    data_shuffle()

    #범주 안에 들어가게 하기 위해
    train_imags = train_images / 255
    
    #cifar_mnist = datasets.cifar10
    #(train_images, train_labels), (test_images, test_labels) = cifar_mnist.load_data()
    #print(test_images.shape)
    '''
    train_images = np.array(train_images, dtype = np.float32)
    train_images = train_images / 255

    test_images = np.array(test_images, dtype = np.float32)
    test_images = test_images / 255

    train_labels = utils.to_categorical(train_labels, num_classes)
    test_labels = utils.to_categorical(test_labels, num_classes)

    print('-------------------------------------------------')
    print(train_labels.shape)
    '''
    print('\n\n\nTraining Start\n\n\n')
    #one-hot 인코딩을 위해 범주형 변수를 변환시킨다
    train_labels = utils.to_categorical(train_labels, num_classes)
    #훈련 시작
    split_data_and_train_validate(train_images, train_labels)
    

#image 의 사이즈를 구한다.
def data_img_size():
    global MINIMUM_IMAGE_NUM
    width = 0
    height = 0
    total_num = 0

    for class_folder in class_names:
        for image_seq in range(MINIMUM_IMAGE_NUM):
            filename = data_folder + class_folder + '/' + class_folder + str(image_seq) + '.jpg' 
            if os.path.exists(filename) == False:
                continue
            img = cv2.imread(filename)
            img = list(img)
            width += len(img)
            height += len(img[0])
            total_num += 1
    
    width = width / total_num
    height = height / total_num
    width = int(width)
    height = int(height)
    return (width, height)

def test_data_img(width, height):
    global test_images

    for class_folder in test_names:
        for image_seq in range(2):
            filename = class_folder + '/' + class_folder + str(image_seq) + '.jpg' 
            if os.path.exists(filename) == False:
                continue
            img = cv2.imread(filename)
            img = cv2.resize(img, dsize=(width, height), interpolation=cv2.INTER_AREA)
            img = list(img)
            test_images.append(img)
    
    test_images = np.array(test_images)

#train data를 넣는다.
def train_data_img(width, height):
    global train_images
    global train_labels
    global MINIMUM_IMAGE_NUM

    for class_folder in class_names:
        for image_seq in range(MINIMUM_IMAGE_NUM):
            filename = data_folder + class_folder + '/' + class_folder + str(image_seq) + '.jpg' 
            if os.path.exists(filename) == False:
                continue
            img = cv2.imread(filename)
            img = cv2.resize(img, dsize=(width, height), interpolation=cv2.INTER_AREA)
            img = list(img)
            train_images.append(img)
            tmp = []
            tmp.append(dict_class_names[class_folder])
            train_labels.append(tmp)
    
    train_images = np.array(train_images)
    train_labels = np.array(train_labels)

#순서대로 데이터 집합을 나눌때, random하게 데이터가 들어가게 하기 위해 shuffle 한다.
def data_shuffle():
    global train_images
    global train_labels

    for_shuffle = np.arange(len(train_images))
    np.random.shuffle(for_shuffle)

    train_images = train_images[for_shuffle]
    train_labels = train_labels[for_shuffle]

#train_data, validate_data, test_data를 random 하게 나눈다.
def split_data_and_train_validate(train_images, train_labels):
    global batch_size
    global global_accuracy
    global final_model
    global model
    #batch_size로 나눌 수 있는 전체 data_set 개수
    data_set_len = int(len(train_images) / batch_size)
    print(data_set_len)
    #valid data set을 놓는다.
    for valid in range(0, data_set_len):
        valid_data = train_images[valid * batch_size : (valid + 1) * batch_size]
        valid_label = train_labels[valid * batch_size : (valid + 1) * batch_size]
        train_data_set = []
        train_data_label = []
        #train data set을 모아놓는다.
        for train in range(0, data_set_len):
            if valid == train:
                continue
            train_data_set.append(train_images[train * batch_size : (train + 1) * batch_size])
            train_data_label.append(train_labels[train * batch_size : (train + 1) * batch_size])
        
        #train 함수 호출
        train_model(train_data_set, train_data_label)
        #validate 함수 호출
        accuracy = validate(valid_data, valid_label)
        #모델의 정확도가 높으면, 이 모델을 저장한다.
        print('accuracy : ' + str(accuracy))
        if accuracy >= global_accuracy:
            final_model = model
            model_save()
            global_accuracy = accuracy
        model = []
        break


def train_model(train_data_set, train_data_label):
    global model
    #train data set을 가지고 훈련시킨다.
    for idx, train_data in enumerate(train_data_set):
        batch_model = Model(str(idx))
        batch_model.make_model(train_data)
        batch_model.model_compile()
        batch_model.model_fit(train_data, train_data_label[idx])
        model.append(batch_model)

def output_result(test_images):
    global final_model
    global class_names

    predictions = np.zeros(1 * num_classes).reshape(1, num_classes)

    for idx, md in enumerate(final_model):
        p = md.predictions(test_images)
        predictions += p

    print(class_names[tf.compat.v1.argmax(predictions, 1)[0]])

def validate(valid_data, valid_label):
    #accuracy를 측정하여 정확도를 비교한다.
    global model
    valid_size = len(valid_label)
    predictions = np.zeros(valid_size * num_classes).reshape(valid_size, num_classes)

    for idx, md in enumerate(model):
        p = md.predictions(valid_data)
        predictions += p
    
    #gpu를 사용하기 위한 코드
    #backend의 k에서 훈련시켜온 session을 tf에 입력시킨다
    #tf를 통해 앙상블 시킨 모든 모델에 대한 정확도를 측정한다.
    with tf.device("/gpu:0"):
        with tf.compat.v1.Session(graph = K.get_session().graph) as sess:
            ensemble_correct_prediction = tf.compat.v1.equal(
                tf.compat.v1.argmax(predictions, 1), tf.compat.v1.argmax(valid_label, 1))
            ensemble_accuracy = tf.compat.v1.reduce_mean(
                tf.compat.v1.cast(ensemble_correct_prediction, tf.compat.v1.float32))
            return sess.run(ensemble_accuracy)

#model의 weights를 저장한다.
def model_save():
    global model
    global checkpoint_dir
    global checkpoint_path
    #전체 모델의 가중치를 저장한다.
    for idx, md in enumerate(final_model):
        md.model.save_weights(checkpoint_dir + checkpoint_path + str(idx))
    
    #model를 reuse하기 위해 초기화 한다.
    model = []

class Model():
    global num_classes
    global epochs
    global checkpoint_dir
    global checkpoint_path

    def __init__(self, name):
        self.name = name
        self.num_classes = num_classes

    def make_model(self, train_images, drop_out = 0.25):
        #전체 모델의 shpae을 정의 해놓는다.
        self.model = keras.Sequential([
            Conv2D(32, kernel_size = (3, 3), padding = 'same', input_shape = train_images.shape[1:], activation = tf.nn.relu),
            MaxPooling2D(pool_size = (2, 2)),
            Dropout(drop_out),

            Conv2D(64, kernel_size = (3, 3), padding = 'same', activation = tf.nn.relu),
            MaxPooling2D(pool_size = (2, 2)),
            Dropout(drop_out),

            Flatten(),
            Dense(64, activation = tf.nn.relu),
            Dropout(drop_out),
            Dense(self.num_classes, activation = tf.nn.softmax)
        ])
    
    def model_compile(self):
        #모델의 shape를 그 shape대로 compile한다.
        if self.model == None:
            print('There is no model')
            return
        
        self.model.compile(
            loss = 'categorical_crossentropy',
            optimizer = 'adam',
            metrics = ['accuracy']
        )
    
    def model_fit(self, train_images, train_labels):
        #latest = tf.compat.v1.train.latest_checkpoint('training_1')
        latest = None
        if latest != None:
            #저장해 놨던 가중치를 load시킨다
            self.model.load_weights(latest)
        else:
            #저장해놓은 가중치가 없으면 훈련시킨다
            #callback 함수들 정의
            self.early_stopping = EarlyStopping(monitor = 'val_loss', patience = 10)
            #self.check_point = tf.compat.v1.keras.callbacks.ModelCheckpoint(filepath = checkpoint_path, save_weights_only = True, verbose = 1, period = 5)
            #gpu로 돌리기 위한 코드
            #코드 실행안되면 tensorflow-gpu 버전에 맞춰 CUDA Toolkit 설치 후 확인
            with tf.device('/gpu:0'):
                self.history = self.model.fit(
                    train_images,
                    train_labels,
                    epochs = epochs,
                    validation_data = (train_images, train_labels),
                    shuffle = True,
                    callbacks = [self.early_stopping]
                )

    def predictions(self, test_images):
        #실제 데이터를 입력하여 예측한다.
        return self.model.predict(test_images)
    
    def description(self):
        if self.model == None:
            print('There is no model')
            return
        #모델 전체 shape에 대한 기술을 보여준다.
        print(self.model.summary())

    #이미 훈련된 가중치를 load 시킨다.
    def load_weights():
        self.model.load_weights(checkpoint_dir + checkpoint_path + str(self.name))


if __name__ == "__main__":
    #global global_accuracy
    load_data()
    output_result(test_images)
'''
for i in range(0, total_image_num):
    batch_model = Model(str(i))
    split_train_image = train_images[i * batch_size : (i + 1) * batch_size, :]
    split_train_labels = train_labels[i * batch_size : (i + 1) * batch_size, :]
    batch_model.make_model(split_train_image)
    batch_model.model_compile()
    batch_model.model_fit(split_train_image, split_train_labels)
    model.append(batch_model)

test_size = len(test_images)
predictions = np.zeros(test_size * num_classes).reshape(test_size, num_classes)

for i in range(0, total_image_num):
    p = model[i].predictions(test_images)
    predictions += p

#backend의 k에서 훈련시켜온 session을 tf에 입력시킨다
#tf를 통해 앙상블 시킨 모든 모델에 대한 정확도를 측정한다.
with tf.compat.v1.Session(graph = K.get_session().graph) as sess:
    ensemble_correct_prediction = tf.compat.v1.equal(
        tf.compat.v1.argmax(predictions, 1), tf.compat.v1.argmax(test_labels, 1))
    ensemble_accuracy = tf.compat.v1.reduce_mean(
        tf.compat.v1.cast(ensemble_correct_prediction, tf.compat.v1.float32))
    print('Ensemble accuracy : ', sess.run(ensemble_accuracy))
'''




