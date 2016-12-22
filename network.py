from keras.models import Sequential
from keras.layers import Dense, Conv2D, BatchNormalization, Activation, ZeroPadding2D, MaxPooling2D, Flatten, Dropout
from keras.callbacks import TensorBoard, ModelCheckpoint
from keras.preprocessing.image import ImageDataGenerator
import tensorflow as tf
tf.python.control_flow_ops = tf # hack for compatability with tf .11

def VGG_M():
    model = Sequential(
        [Conv2D(96,7,7,
                subsample=(2,2),
                activation='relu', input_shape=[256,256,3]),
         BatchNormalization(),

         MaxPooling2D((2,2)),
         ZeroPadding2D((1,1)),
         Conv2D(256,5,5,
                subsample=(2,2),
                activation='relu'),
         BatchNormalization(),
         MaxPooling2D((2,2)),
         ZeroPadding2D((1,1)),
         Conv2D(512,3,3,activation='relu'),
         ZeroPadding2D((1,1)),
         Conv2D(512,3,3,activation='relu'),
         ZeroPadding2D((1,1)),
         Conv2D(512,3,3,activation='relu'),

         BatchNormalization(),
         MaxPooling2D((2,2)),

         Flatten(),
         Dense(4096, activation='relu'),
         BatchNormalization(),
         Dropout(.5),
         Dense(4096, activation='relu'),
         Dropout(.5),
         BatchNormalization(),

         Dense(1000, activation='softmax')
        ]
    )
    return model

from os.path import expanduser, join

def get_generators(datapath = expanduser('~/Datasets/ImageNet/raw-data/'), batch_size=16):
    traindir = join(datapath, 'train')
    trainpp = ImageDataGenerator(horizontal_flip=True, rotation_range=30)
    traingen = trainpp.flow_from_directory(traindir, batch_size=batch_size)
    
    validdir = join(datapath, 'validation')    
    validpp = ImageDataGenerator(horizontal_flip=True)
    validgen = validpp.flow_from_directory(validdir, batch_size=batch_size)

    return traingen, validgen

def train():
    mc = ModelCheckpoint(filepath="models", verbose=1, save_best_only=True)
    tb = TensorBoard("logs")
    train_gen, valid_gen = get_generators()
    train_size = 1281167
    valid_size = 50000
    m = VGG_M()
    m.compile(optimizer='adam', loss='categorical_crossentropy')
    m.fit_generator(train_gen, train_size, 1000, callbacks = [mc,tb],
                    validation_data=valid_gen, nb_val_samples=valid_size/5, nb_worker=8, pickle_safe=True)
    
if __name__ == '__main__':
    train()
