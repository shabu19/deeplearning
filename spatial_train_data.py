import csv
import os.path
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.utils import to_categorical

class DataSet():
    def __init__(self, num_of_snip=5, opt_flow_len=10, image_shape=(224, 224), class_limit=None):
        self.opt_flow_len = opt_flow_len
        self.num_of_snip = num_of_snip
        self.class_limit = class_limit
        self.image_shape = image_shape

        # Get the data.
        self.data_list = self.get_data_list()

        # Get the classes.
        self.classes = self.get_classes()

        # Now do some minor data cleaning
        self.data_list = self.clean_data_list()

    @staticmethod
    def get_data_list():
        with open(os.path.join('C:/Users/User/Documents/PyCharm_Projects/two-stream-action-recognition-keras-master', 'data_list.csv'), 'r') as fin:
            reader = csv.reader(fin)
            data_list = list(reader)

        return data_list

    def clean_data_list(self):
        data_list_clean = []
        for item in self.data_list:
            if item[1] in self.classes:
                data_list_clean.append(item)

        return data_list_clean

    def get_classes(self):
        classes = []
        for item in self.data_list:
            if item[1] not in classes:
                classes.append(item[1])

        # Sort them.
        classes = sorted(classes)

        # Return.
        if self.class_limit is not None:
            return classes[:self.class_limit]
        else:
            return classes

    def get_class_one_hot(self, class_str):
        # Encode it first.
        label_encoded = self.classes.index(class_str)

        # Now one-hot it.
        label_hot = to_categorical(label_encoded, len(self.classes))

        return label_hot

def get_generators(data, image_shape=(224, 224), batch_size=128):
    train_datagen = ImageDataGenerator(rescale=1./255, shear_range=0.2, horizontal_flip=True, rotation_range=10., width_shift_range=0.2, height_shift_range=0.2)

    test_datagen = ImageDataGenerator(rescale=1./255)

    train_generator = train_datagen.flow_from_directory(os.path.join('C:/Users/User/Documents/PyCharm_Projects/datasets/ucf-101/static', 'train'), target_size=image_shape, batch_size=batch_size, classes=data.classes, class_mode='categorical')

    validation_generator = test_datagen.flow_from_directory(os.path.join('C:/Users/User/Documents/PyCharm_Projects/datasets/ucf-101/static', 'test'), target_size=image_shape, batch_size=batch_size, classes=data.classes, class_mode='categorical')

    return train_generator, validation_generator

