from keras.models import Sequential
from keras.layers import Dense, Conv2D, BatchNormalization, Activation, ZeroPadding2D, MaxPooling2D, Flatten, Dropout

def VGG_M():
    model = Sequential(
        [Conv2D(96,7,7,
                subsample=(2,2),
                activation='relu', input_shape=[224,224,3]),
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
