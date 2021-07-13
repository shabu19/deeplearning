
from analyze_model import plot_results
from temporal_train_model import ResearchModels
from temporal_train_data import DataSet
from callbacks_temporal import callbacks
import numpy as np
from tensorflow.keras.optimizers import SGD, Adam
from sklearn.metrics import classification_report, confusion_matrix

def train(num_of_snip=5, opt_flow_len=10, saved_model=None, class_limit=None, image_shape=(224, 224), batch_size=32, nb_epoch=100):

    # callbacks
    cb = callbacks('cp-temporal', 'tb-temporal', 'logs-temporal')

    # Get the data and process it.
    data = DataSet(num_of_snip=num_of_snip, opt_flow_len=opt_flow_len, image_shape=image_shape, class_limit=class_limit)

    steps_per_epoch = (len(data.data_list) * 0.95) // batch_size

    generator = data.stack_generator(batch_size, 'train')
    val_generator = data.stack_generator(batch_size, 'test')

    temporal_cnn = ResearchModels(nb_classes=len(data.classes), num_of_snip=num_of_snip, opt_flow_len=opt_flow_len, image_shape=image_shape, saved_model=saved_model)

    # Fit!
    temporal_cnn.model.fit(x=generator, steps_per_epoch=steps_per_epoch, epochs=nb_epoch, verbose=1, callbacks=cb, validation_data=val_generator, validation_steps=1, max_queue_size=20, workers=1, use_multiprocessing=False)


def main():
    saved_model = None #'C:/Users/User/Desktop/205.hdf5'
    class_limit = 101   # int, can be 1-101 or None
    num_of_snip = 1     # number of chunks used for each video
    opt_flow_len = 10   # number of optical flow frames used
    image_shape=(224, 224)
    batch_size = 128
    nb_epoch = 500

    train(num_of_snip=num_of_snip, opt_flow_len=opt_flow_len, saved_model=saved_model, class_limit=class_limit, image_shape=image_shape, batch_size=batch_size, nb_epoch=nb_epoch)

if __name__ == '__main__':
    main()