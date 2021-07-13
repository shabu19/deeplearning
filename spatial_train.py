from spatial_train_model import get_model, freeze_all_but_mid_and_top, freeze_all_but_top
from spatial_train_data import DataSet, get_generators
from callbacks_spatial import callbacks
from analyze_model import plot_results

def train(class_limit=None, image_shape=(224, 224), batch_size=32, nb_epoch=100, saved_weights=None):

    # callbacks
    cb = callbacks('cp-spatial', 'tb-spatial', 'logs-spatial')

    # get data list
    data = DataSet(image_shape=image_shape, class_limit=class_limit)

    # get generators.
    train_generator, val_generator = get_generators(data=data, image_shape=image_shape, batch_size=batch_size)

    # get the model.
    model = get_model(data=data)

    if saved_weights is None:
        model = model
    else:
        model.load_weights(saved_weights)

    # compile and train
    model = freeze_all_but_top(model)
    model.fit_generator(train_generator, steps_per_epoch=100, validation_data=val_generator, validation_steps=10, epochs=10, callbacks=[])

    # compile and train
    model = freeze_all_but_mid_and_top(model)
    history = model.fit_generator(train_generator, steps_per_epoch=100, validation_data=val_generator, validation_steps=10, epochs=nb_epoch, callbacks=cb)

def main():
    saved_weights = None #'C:/Users/User/Desktop/spatiaol/092.hdf5'
    class_limit = 100
    image_shape = (224, 224)
    batch_size = 128
    nb_epoch = 200

    train(class_limit=class_limit, image_shape=image_shape, batch_size=batch_size, nb_epoch=nb_epoch, saved_weights=saved_weights)

if __name__ == '__main__':
    main()
