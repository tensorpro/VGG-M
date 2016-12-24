from network import get_generators, VGG_M
from os.path import join, expand_user, lisdir
base_path = expanduser('~/Datasets/ImageNet/raw-data/validation')
from keras.preprocessing.image import ImageDataGenerator

def predict():
    m = VGG_M()
    dg = ImageDataGenerator()
    for sysnet in listdir('.')[:3]:
        dg.flow_from_directory('.', batch_size=50, classes=[synset])
    preds = m.predict_generator(valid_gen, 50)
    print preds
    # m.fit_generator(train_gen, 1000, 1000, callbacks = [tb, pb],
    #                 validation_data=valid_gen, nb_val_samples=500, nb_worker=16, pickle_safe=True)
