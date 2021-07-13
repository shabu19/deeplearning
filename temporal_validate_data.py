import csv
import numpy as np
import os.path
from tensorflow.keras.utils import to_categorical
import cv2

class DataSet():
    def __init__(self, class_limit=None, image_shape=(224, 224), original_image_shape=(341, 256), n_snip=5, opt_flow_len=10, batch_size=16):
        self.class_limit = class_limit
        self.image_shape = image_shape
        self.original_image_shape = original_image_shape
        self.n_snip = n_snip
        self.opt_flow_len = opt_flow_len
        self.batch_size = batch_size

        self.opt_flow_path = os.path.join('C:/Users/User/Documents/PyCharm_Projects/datasets/ucf-101', 'optical-flow')

        # Get the data.
        self.data_list = self.get_data_list()

        # Get the classes.
        self.classes = self.get_classes()

        # Now do some minor data cleaning
        self.data_list = self.clean_data_list()

        # Get the right dataset for the generator.
        train, test = self.split_train_test()
        self.data_list = test
        
        # number of batches in 1 epoch
        self.n_batch = len(self.data_list) // self.batch_size

    @staticmethod
    def get_data_list():
        with open(os.path.join('C:/Users/User/Documents/PyCharm_Projects/two-stream-action-recognition-keras-master/', 'data_list_org.csv'), 'r') as fin:
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

        classes = sorted(classes)
        if self.class_limit is not None:
            return classes[:self.class_limit]
        else:
            return classes

    def get_class_one_hot(self, class_str):
        # Encode it first.
        label_encoded = self.classes.index(class_str)

        # Now one-hot it.
        label_hot = to_categorical(label_encoded, len(self.classes))

        assert label_hot.shape[0] == len(self.classes)

        return label_hot

    def split_train_test(self):
        train = []
        test = []
        for item in self.data_list:
            if item[0] == 'train':
                train.append(item)
            else:
                test.append(item)
        return train, test

    def validation_generator(self):

        print("\nCreating validation generator with %d samples.\n" % len(self.data_list))

        idx = 0
        while 1:
            idx = idx % self.n_batch
            print("\nGenerating batch number {0}/{1} ...".format(idx + 1, self.n_batch))
            idx += 1
            
            X_batch = []
            y_batch = []

            # Get a list of batch-size samples.
            batch_list = self.data_list[idx * self.batch_size: (idx + 1) * self.batch_size]

            for row in batch_list:
                # Get the stacked optical flows from disk.
                X = self.get_stacked_opt_flows(row)
                
                # Get the corresponding labels
                y = self.get_class_one_hot(row[1])
                y = np.array(y)
                y = np.squeeze(y)

                X_batch.append(X)
                y_batch.append(y)

            X_batch = np.array(X_batch)
            y_batch = np.array(y_batch)

            if len(X_batch) >0:
                yield X_batch, y_batch

    def get_stacked_opt_flows(self, row):

        output = []
        opt_flow_dir_x = os.path.join(self.opt_flow_path, 'u', row[1], row[2])
        opt_flow_dir_y = os.path.join(self.opt_flow_path, 'v', row[1], row[2])

        # spatial parameters
        left = int((self.original_image_shape[0] - self.image_shape[0]) * 0.5)
        top = int((self.original_image_shape[1] - self.image_shape[1]) * 0.5)
        right = left + self.image_shape[0]
        bottom = top + self.image_shape[1]
            
        # temporal parameters
        total_frames = len(os.listdir(opt_flow_dir_x))
        if total_frames - self.opt_flow_len + 1 < self.n_snip:
            loop = True
            start_frame_window_len = 1
        else:
            loop = False
            start_frame_window_len = (total_frames - self.opt_flow_len + 1) // self.n_snip

        # loop over snippets
        for i_snip in range(self.n_snip):
            if loop:
                start_frame = i_snip % (total_frames - self.opt_flow_len + 1) + 1
            else:
                start_frame = int(0.5 * start_frame_window_len + 0.5) + start_frame_window_len * i_snip

            # Get the optical flow stack
            frames = range(start_frame, start_frame + self.opt_flow_len)
            opt_flow_stack = []
            # loop over frames
            for i_frame in frames:
                # horizontal components
                img = None
                img = cv2.imread(opt_flow_dir_x + '/frame' + "%06d" %i_frame + '.jpg', 0)
                img = np.array(img)
                img = img - np.mean(img)
                img = img[top: bottom, left: right]
                img = img / 255.0
                opt_flow_stack.append(img)
    
                # vertical components
                img2 = None
                img2 = cv2.imread(opt_flow_dir_y + '/frame' + "%06d" %i_frame + '.jpg', 0)
                img2 = np.array(img2)
                img2 = img2 - np.mean(img2)
                img2 = img2[top: bottom, left: right]
                img2 = img2 / 255.0
                opt_flow_stack.append(img2)

            opt_flow_stack = np.array(opt_flow_stack)
            opt_flow_stack = np.swapaxes(opt_flow_stack, 0, 2)

            output.append(opt_flow_stack)

        output = np.array(output)

        return output

