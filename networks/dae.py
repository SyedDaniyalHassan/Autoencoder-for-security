from tensorflow import keras
from keras.datasets import cifar10
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from tensorflow.keras.layers import Conv2D, Input, Dense, Reshape, Conv2DTranspose,Activation, BatchNormalization, ReLU, Concatenate
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import ModelCheckpoint
from keras.models import Sequential , load_model

class Dae :
    def __init__(self,classifier= 'resnet',epochs=200, batch_size=128, load_weights=True):
        self.name               = 'DAE'
        self.model_filename     = 'networks/models/Dae_best_1.h5'
        self.num_classes        = 10
        self.input_shape        = 32, 32, 3
        self.batch_size         = batch_size
        self.epochs             = epochs
        self.iterations         = 391
        self.weight_decay       = 0.0001
        self.log_filepath       = r'networks/models/Dae_best/'
        self.classifier_path    = 'networks/models/'+classifier+'.h5'
        self.classifier_name = classifier
        self._model = self.denoising_autoencoder()
        print("hello world")
        try :
            self._model.load_weights(self.model_filename)
            self._classifier_model = load_model(self.classifier_path)
            print("Successfully loaded",self.name)
        except(ImportError, ValueError, OSError):
            print("Model not loaded")
    def count_params(self):
        return self._model.count_params()+self._classifier_model.count_params()


    def deconv_block(self,x, filters, kernel_size):
        x = Conv2DTranspose(filters=filters,
                        kernel_size=kernel_size,
                        strides=2,
                        padding='same')(x)
        x = BatchNormalization()(x)
        x = ReLU()(x)
        return x
    def conv_block(self,x, filters, kernel_size, strides=2):
        x = Conv2D(filters=filters,
                kernel_size=kernel_size,
                strides=strides,
                padding='same')(x)
        x = BatchNormalization()(x)
        x = ReLU()(x)
        return x

    def color_preprocessing(self, x_train, x_test):
        x_train = x_train.astype('float32')
        x_test = x_test.astype('float32')
        mean = [125.307, 122.95, 113.865]
        std  = [62.9932, 62.0887, 66.7048]
        for i in range(3):
            x_train[:,:,:,i] = (x_train[:,:,:,i] - mean[i]) / std[i]
            x_test[:,:,:,i] = (x_test[:,:,:,i] - mean[i]) / std[i]
        return x_train, x_test


    def denoising_autoencoder(self):
        dae_inputs = Input(shape=(32, 32, 3), name='dae_input')
        conv_block1 = self.conv_block(dae_inputs, 32, 3)
        conv_block2 = self.conv_block(conv_block1, 64, 3)
        conv_block3 = self.conv_block(conv_block2, 128, 3)
        conv_block4 = self.conv_block(conv_block3, 256, 3)
        conv_block5 = self.conv_block(conv_block4, 256, 3, 1)

        deconv_block1 = self.deconv_block(conv_block5, 256, 3)
        merge1 = Concatenate()([deconv_block1, conv_block3])
        deconv_block2 = self.deconv_block(merge1, 128, 3)
        merge2 = Concatenate()([deconv_block2, conv_block2])
        deconv_block3 = self.deconv_block(merge2, 64, 3)
        merge3 = Concatenate()([deconv_block3, conv_block1])
        deconv_block4 = self.deconv_block(merge3, 32, 3)
        final_deconv = Conv2DTranspose(filters=3,kernel_size=3,padding='same')(deconv_block4)
        dae_outputs = Activation('sigmoid', name='Dae_output')(final_deconv)
        
        return Model(dae_inputs, dae_outputs, name='Dae')


    def scheduler(self, epoch):
        if epoch <= 60:
            return 0.05
        if epoch <= 120:
            return 0.01
        if epoch <= 160:    
            return 0.002
        return 0.0004
    def color_process(self, imgs):
        if imgs.ndim < 4:
            imgs = np.array([imgs])
        imgs = imgs.astype('float32')
        mean = [125.307, 122.95, 113.865] #[0, 0, 0]
        std  = [62.9932, 62.0887, 66.7048]#[255, 255, 255]
        for img in imgs:
            for i in range(3):
                img[:,:,i] = (img[:,:,i] - mean[i]) / std[i]
        return imgs
    def predict(self, img):
        processed = self.color_process(img)
        remove = self._model.predict(processed, batch_size=self.batch_size)
        return self._classifier_model.predict(remove , batch_size=self.batch_size)
    
    def predict_one(self, img):
        return self.predict(img)[0]

    def accuracy(self):
        (x_train, y_train), (x_test, y_test) = cifar10.load_data()
        y_train = keras.utils.to_categorical(y_train, self.num_classes)
        y_test = keras.utils.to_categorical(y_test, self.num_classes)

        # color preprocessing
        x_train, x_test = self.color_preprocessing(x_train, x_test)

        return self._classifier_model.evaluate(x_test, y_test, verbose=0)[1]