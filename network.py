from keras.models import Sequential
from keras.layers import Dense, Conv2D, BatchNormalization, Activation, ZeroPadding2D, MaxPooling2D, Flatten, Dropout
from keras.layers import Lambda
from keras.callbacks import TensorBoard, ModelCheckpoint, ProgbarLogger
from keras.preprocessing.image import ImageDataGenerator
import tensorflow as tf

tf.python.control_flow_ops = tf # hack for compatability with tf .11

md = [107,107,3]
imn = [224,224,3]
imn = [256,256,3]

# .lrn(2, 0.0001, 0.75, name='norm1')
def lrn(input, radius=2, alpha=.001, beta=.75, name='LRN', bias=1.0):
        return tf.nn.local_response_normalization(input,
                                                  depth_radius=radius,
                                                  alpha=alpha,
                                                  beta=beta,
                                                  bias=bias,
                                                  name=name)


def convW(W, num):
    Wb = W['conv{}'.format(num)]
    
    

def VGGNW():
    model = Sequential(
        [Conv2D(96,7,7, subsample=(2,2),input_shape=imn),
         Activation('relu'),
         #LRN
         Lambda(lrn),
         MaxPooling2D((3,3),(2,2)),

         Conv2D(256,5,5, subsample=(2,2)),
         Activation('relu'),
         #LRN
         Lambda(lrn),
         MaxPooling2D((2,2)),
         
         Conv2D(512,3,3,activation='relu'),
         ZeroPadding2D((1,1)),
         Conv2D(512,3,3,activation='relu'),
         ZeroPadding2D((1,1)),
         Conv2D(512,3,3,activation='relu'),
         MaxPooling2D((2,2)),

         BatchNormalization(),
        

         Flatten(),
         Dense(4096, activation='relu'),
         BatchNormalization(),
         Dropout(.5),
         Dense(4096, activation='relu'),
         Dropout(.5),
         BatchNormalization(),

         Dense(1000, activation='softmax'),
        ]
    )
    model.load_weights('VGGM.h5')
    return model


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
    pb = ProgbarLogger()
    train_gen, valid_gen = get_generators()
    train_size = 1281167
    valid_size = 50000
    m = VGG_M()
    m.compile(optimizer='adam', loss='categorical_crossentropy', metrics = ['top_k_categorical_accuracy',
                                                                            'categorical_accuracy'])
    m.fit_generator(train_gen, train_size/2000, 1000, callbacks = [mc,tb, pb],
                    validation_data=valid_gen, nb_val_samples=valid_size/5, nb_worker=16, pickle_safe=True)
    
if __name__ == '__main__':
    train()
