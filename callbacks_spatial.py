import os

from tensorflow.keras.callbacks import TensorBoard, ModelCheckpoint, EarlyStopping, CSVLogger

def callbacks(cp, tb, logs):

    # Callbacks: Save the model.
    directory1 = os.path.join('out', 'checkpoints', cp)
    if not os.path.exists(directory1):
        os.makedirs(directory1)
    checkpointer = ModelCheckpoint(filepath=os.path.join(directory1, '{epoch:03d}.hdf5'), monitor='val_accuracy', verbos=1, save_best_only=True, save_weights_only=True, mode='max')

    # Callbacks: TensorBoard
    directory2 = os.path.join('out', 'tensorboard', tb)
    if not os.path.exists(directory2):
        os.makedirs(directory2)
    tensorboard = TensorBoard(log_dir=os.path.join(directory2))

    # Callbacks: Early Stopper
    early_stopper = EarlyStopping(monitor='loss', patience=100)

    # Callbacks: Save Results.
    directory3 = os.path.join('out', 'results', logs)
    if not os.path.exists(directory3):
        os.makedirs(directory3)
    csv_logger = CSVLogger(os.path.join(directory3, 'training-' + 'logs' + '.csv'))

    cb = [checkpointer, tensorboard, early_stopper, csv_logger]

    return cb


