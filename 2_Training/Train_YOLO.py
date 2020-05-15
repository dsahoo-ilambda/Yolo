"""
MODIFIED FROM keras-yolo3 PACKAGE, https://github.com/qqwweee/keras-yolo3
Retrain the YOLO model for your own dataset.
"""

import os
import sys
import argparse
import subprocess
import pandas as pd


def create_pred_df(out_df_path, source_df_path, classes):
    out_df = pd.read_csv(out_df_path)
    source_df = pd.read_csv(source_df_path, index_col='index')
    out_df['predicted_label'] = out_df['label'].apply(lambda x: classes[x])
    out_df['true_label'] = out_df['image'].apply(lambda x: x.split('.')[0].split('_')[-1])
    df = pd.merge(out_df, source_df, left_on='image', right_on='new_name')
    df['iou'] = df.apply(iou, axis=1)

    df['max_confidence'] = df.groupby('image')['confidence'].transform(max)
    # print(df.shape)
    return df, source_df


def get_accuracies(pred_df, source_df):
    print(f"Total length of test set: {len(source_df)}")
    print(f"Total unique detections: {len(pred_df['image'].unique())}")
    top1accuracy(pred_df, source_df)
    topnaccuracy(pred_df, source_df)
    avg_iou = np.mean(pred_df[pred_df.confidence == pred_df.max_confidence]['iou'])
    print(f"Avg IOU with Truth BB: {avg_iou: 0.3f} ")


def top1accuracy(pred_df, source_df):
    top1_pred = len(pred_df[(pred_df.confidence == pred_df.max_confidence) & (pred_df.predicted_label == pred_df.true_label)])
    top1_accuracy = top1_pred/len(source_df)
    print(f"top-1 accuracy = {top1_accuracy}")


def topnaccuracy(pred_df, source_df):
    xx = pred_df.groupby('image')['predicted_label'].apply(list).reset_index()
    xx['true_label'] = xx['image'].apply(lambda x: x.split('.')[0].split('_')[-1])
    topn_pred = sum(xx.apply(lambda row: row['true_label'] in set(row['predicted_label']), axis=1))
    print(f"top-n accuracy = {topn_pred/len(source_df)}")


def iou(row):
    true_x1 = row['new_x_1']
    true_y1 = row['new_y_1']
    true_x2 = row['new_x_2']
    true_y2 = row['new_y_2']
    pred_x1 = row['xmin']
    pred_y1 = row['ymin']
    pred_x2 = row['xmax']
    pred_y2 = row['ymax']
    inner_xmin = max(min(true_x1, true_x2), min(pred_x1, pred_x2))
    inner_xmax = min(max(true_x1, true_x2), max(pred_x1, pred_x2))
    inner_ymin = max(min(true_y1, true_y2), min(pred_y1, pred_y2))
    inner_ymax = min(max(true_y1, true_y2), max(pred_y1, pred_y2))
    # print(inner_xmin, inner_ymin, inner_xmax, inner_ymax)
    h1 = abs(true_y1 - true_y2)
    w1 = abs(true_x1 - true_x2)
    h2 = abs(pred_y1 - pred_y2)
    w2 = abs(pred_x1 - pred_x2)
    innerh = abs(inner_ymax - inner_ymin)
    innerw = abs(inner_xmax - inner_xmin)
    area_inner = innerh * innerw
    iou = (area_inner) / (h1 * w1 + h2 * w2 - area_inner)
    return (iou)


def get_parent_dir(n=1):
    """ returns the n-th parent dicrectory of the current
    working directory """
    current_path = os.path.dirname(os.path.abspath(__file__))
    for k in range(n):
        current_path = os.path.dirname(current_path)
    return current_path


src_path = os.path.join(get_parent_dir(0), 'src')
tiny_yolo_root = '/home/ilambda/goods_viewer/Debasish/tiny_yolo'
sys.path.append(src_path)

utils_path = os.path.join(get_parent_dir(1), 'Utils')
sys.path.append(utils_path)

import numpy as np
import tensorflow as tf

TF_VERSION2 = tf.__version__.startswith("2")
if TF_VERSION2: from tensorflow import keras
import keras.backend as K
from keras.layers import Input, Lambda
from keras.models import Model
from keras.optimizers import Adam
from keras.callbacks import TensorBoard, ModelCheckpoint, ReduceLROnPlateau, EarlyStopping
from keras_yolo3.yolo3.model import preprocess_true_boxes, yolo_body, tiny_yolo_body, yolo_loss
from keras_yolo3.yolo3.utils import get_random_data
from PIL import Image
from time import time
import pickle

from Train_Utils import get_classes, get_anchors, create_model, create_tiny_model, data_generator, \
    data_generator_wrapper, ChangeToOtherMachine

keras_path = os.path.join(src_path, 'keras_yolo3')
Data_Folder = os.path.join(get_parent_dir(1), 'Data')
# Image_Folder = os.path.join(Data_Folder,'Source_Images','Training_Images')
# VoTT_Folder = os.path.join(Image_Folder,'vott-csv-export')
YOLO_filename = os.path.join(tiny_yolo_root, 'training_labels.txt')

# Model_Folder = os.path.join(Data_Folder, 'Model_Weights')
YOLO_classname = os.path.join(tiny_yolo_root, 'category_names.txt')

log_dir = os.path.join(tiny_yolo_root, 'models')
anchors_path = os.path.join(tiny_yolo_root, 'yolo-tiny_anchors.txt')
weights_path = os.path.join(tiny_yolo_root, 'darknet-tiny-yolo.h5')

FLAGS = None

if __name__ == '__main__':
    # Delete all default flags
    parser = argparse.ArgumentParser(argument_default=argparse.SUPPRESS)
    '''
    Command line options
    '''

    parser.add_argument(
        "--annotation_file", type=str, default=YOLO_filename,
        help="Path to annotation file for Yolo. Default is " + YOLO_filename
    )
    parser.add_argument(
        "--classes_file", type=str, default=YOLO_classname,
        help="Path to YOLO classnames. Default is " + YOLO_classname
    )

    parser.add_argument(
        "--log_dir", type=str, default=log_dir,
        help="Folder to save training logs and trained weights to. Default is " + log_dir
    )

    parser.add_argument(
        "--anchors_path", type=str, default=anchors_path,
        help="Path to YOLO anchors. Default is " + anchors_path
    )

    parser.add_argument(
        "--weights_path", type=str, default=weights_path,
        help="Path to pre-trained YOLO weights. Default is " + weights_path
    )
    parser.add_argument(
        "--val_split", type=float, default=0.1,
        help="Percentage of training set to be used for validation. Default is 10 percent."
    )
    parser.add_argument(
        "--is_tiny", default=False, action="store_true",
        help="Use the tiny Yolo version for better performance and less accuracy. Default is False."
    )
    parser.add_argument(
        "--train_step1", default=False, action="store_true",
        help="Train the model with only the detection layer unfrozen. Default is False."
    )
    parser.add_argument(
        "--random_seed", type=int, default=43,
        help="Random seed value to make script deterministic. Default is 43, i.e. non-deterministic."
    )
    parser.add_argument(
        "--epochs", type=float, default=10,
        help="Number of epochs for training last layers and number of epochs for fine-tuning layers. Default is 10."
    )
    parser.add_argument(
        "--lr", type=float, default=0.0001,
        help="Learning rate for step_2: with all layers unfrozen. Default is 0.0001"
    )
    parser.add_argument(
        "--is_augment", default=False, action="store_true",
        help="Enable random train image augmentation Default is False."
    )
    parser.add_argument(
        "--infer", default=False, action="store_true", dest='inter',
        help="Invokes the detector script after the end of the training"
    )

    FLAGS = parser.parse_args()

    np.random.seed(FLAGS.random_seed)

    log_dir = FLAGS.log_dir
    if not os.path.exists(log_dir):
        os.mkdir(log_dir)

    class_names = get_classes(FLAGS.classes_file)
    num_classes = len(class_names)
    anchors = get_anchors(FLAGS.anchors_path)
    weights_path = FLAGS.weights_path

    input_shape = (448, 448)  # multiple of 32, height, width

    epoch1 = FLAGS.epochs if FLAGS.train_step1 else 0
    epoch2 = FLAGS.epochs

    is_tiny_version = (len(anchors) == 6)  # default setting
    if FLAGS.is_tiny:
        model = create_tiny_model(input_shape, anchors, num_classes,
                                  freeze_body=2, weights_path=weights_path)
    else:
        model = create_model(input_shape, anchors, num_classes,
                             freeze_body=2, weights_path=weights_path)  # make sure you know what you freeze

    log_dir_time = os.path.join(log_dir, '{}'.format(int(time())))
    logging = TensorBoard(log_dir=log_dir_time)
    checkpoint = ModelCheckpoint(os.path.join(log_dir, 'checkpoint.h5'),
                                 monitor='val_loss', save_weights_only=True, save_best_only=True, period=5)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=3, verbose=1)
    early_stopping = EarlyStopping(monitor='val_loss', min_delta=0, patience=10, verbose=1)

    val_split = FLAGS.val_split
    with open(FLAGS.annotation_file) as f:
        lines = f.readlines()

    # This step makes sure that the path names correspond to the local machine
    # This is important if annotation and training are done on different machines (e.g. training on AWS)
    # lines  = ChangeToOtherMachine(lines,remote_machine = '')
    np.random.shuffle(lines)
    num_val = int(len(lines) * val_split)
    num_train = len(lines) - num_val

    # Train with frozen layers first, to get a stable loss.
    # Adjust num epochs to your dataset. This step is enough to obtain a decent model.
    if FLAGS.train_step1:
        model.compile(optimizer=Adam(lr=1e-4), loss={
            # use custom yolo_loss Lambda layer.
            'yolo_loss': lambda y_true, y_pred: y_pred})

        batch_size = 32
        print('Train on {} samples, val on {} samples, with batch size {}.'.format(num_train, num_val, batch_size))
        history = model.fit_generator(
            data_generator_wrapper(lines[:num_train], batch_size, input_shape, anchors, num_classes,
                                   random=FLAGS.is_augment),
            steps_per_epoch=max(1, num_train // batch_size),
            validation_data=data_generator_wrapper(lines[num_train:], batch_size, input_shape, anchors, num_classes,
                                                   random=False),
            validation_steps=max(1, num_val // batch_size),
            epochs=epoch1,
            initial_epoch=0,
            callbacks=[logging, checkpoint])
        model.save_weights(os.path.join(log_dir, 'trained_weights_stage_1.h5'))

        step1_train_loss = history.history['loss']

        file = open(os.path.join(log_dir_time, 'step1_loss.npy'), "w")
        with open(os.path.join(log_dir_time, 'step1_loss.npy'), 'w') as f:
            for item in step1_train_loss:
                f.write("%s\n" % item)
        file.close()

        step1_val_loss = np.array(history.history['val_loss'])

        file = open(os.path.join(log_dir_time, 'step1_val_loss.npy'), "w")
        with open(os.path.join(log_dir_time, 'step1_val_loss.npy'), 'w') as f:
            for item in step1_val_loss:
                f.write("%s\n" % item)
        file.close()

    # Unfreeze and continue training, to fine-tune.
    # Train longer if the result is unsatisfactory.
    if True:
        for i in range(len(model.layers)):
            model.layers[i].trainable = True
        model.compile(optimizer=Adam(lr=FLAGS.lr),
                      loss={'yolo_loss': lambda y_true, y_pred: y_pred})  # recompile to apply the change
        print('Unfreeze all layers.')

        batch_size = 4  # note that more GPU memory is required after unfreezing the body
        print('Train on {} samples, val on {} samples, with batch size {}.'.format(num_train, num_val, batch_size))
        history = model.fit_generator(
            data_generator_wrapper(lines[:num_train], batch_size, input_shape, anchors, num_classes,
                                   random=FLAGS.is_augment),
            steps_per_epoch=max(1, num_train // batch_size),
            validation_data=data_generator_wrapper(lines[num_train:], batch_size, input_shape, anchors, num_classes,
                                                   random=False),
            validation_steps=max(1, num_val // batch_size),
            epochs=epoch1 + epoch2,
            initial_epoch=epoch1,
            callbacks=[logging, checkpoint, reduce_lr, early_stopping])
        model_path = os.path.join(log_dir, 'trained_weights_final.h5')
        model.save_weights(model_path)
        print(f"Model saved to : {model_path}")
        step2_train_loss = history.history['loss']

        file = open(os.path.join(log_dir_time, 'step2_loss.npy'), "w")
        with open(os.path.join(log_dir_time, 'step2_loss.npy'), 'w') as f:
            for item in step2_train_loss:
                f.write("%s\n" % item)
        file.close()

        step2_val_loss = np.array(history.history['val_loss'])

        file = open(os.path.join(log_dir_time, 'step2_val_loss.npy'), "w")
        with open(os.path.join(log_dir_time, 'step2_val_loss.npy'), 'w') as f:
            for item in step2_val_loss:
                f.write("%s\n" % item)
        file.close()

    if FLAGS.infer:
        output_dir = os.path.join(log_dir, "inference")
        out_df_path = os.path.join(output_dir, "Detection_Results.csv")
        source_df_path = os.path.join(tiny_yolo_root, "inference", "test_source_df2000.csv")
        if not os.path.exists(output_dir):
            os.mkdir(output_dir)
        command = f"python Detector.py --no_save_image --output {output_dir} --yolo_model {model_path} --classes {FLAGS.classes_file} --anchors {FLAGS.anchors_path}"
        subprocess.call(command, shell=True, cwd=os.path.join(get_parent_dir(1), "3_Inference"))
        out_df, source_df = create_pred_df(out_df_path, source_df_path, class_names)
        get_accuracies(out_df, source_df)


"""
MODIFIED FROM keras-yolo3 PACKAGE, https://github.com/qqwweee/keras-yolo3
Retrain the YOLO model for your own dataset.
"""

import os
import sys
import argparse
import subprocess
import pandas as pd


def create_pred_df(out_df_path, source_df_path, classes):
    out_df = pd.read_csv(out_df_path)
    source_df = pd.read_csv(source_df_path, index_col='index')
    out_df['predicted_label'] = out_df['label'].apply(lambda x: classes[x])
    out_df['true_label'] = out_df['image'].apply(lambda x: x.split('.')[0].split('_')[-1])
    df = pd.merge(out_df, source_df, left_on='image', right_on='new_name')
    df['iou'] = df.apply(iou, axis=1)

    df['max_confidence'] = df.groupby('image')['confidence'].transform(max)
    # print(df.shape)
    return df, source_df


def get_accuracies(pred_df, source_df):
    print(f"Total length of test set: {len(source_df)}")
    print(f"Total unique detections: {len(pred_df['image'].unique())}")
    top1accuracy(pred_df, source_df)
    topnaccuracy(pred_df, source_df)
    avg_iou = np.mean(pred_df[pred_df.confidence == pred_df.max_confidence]['iou'])
    print(f"Avg IOU with Truth BB: {avg_iou: 0.3f} ")


def top1accuracy(pred_df, source_df):
    top1_pred = len(pred_df[(pred_df.confidence == pred_df.max_confidence) & (pred_df.predicted_label == pred_df.true_label)])
    top1_accuracy = top1_pred/len(source_df)
    print(f"top-1 accuracy = {top1_accuracy}")


def topnaccuracy(pred_df, source_df):
    xx = pred_df.groupby('image')['predicted_label'].apply(list).reset_index()
    xx['true_label'] = xx['image'].apply(lambda x: x.split('.')[0].split('_')[-1])
    topn_pred = sum(xx.apply(lambda row: row['true_label'] in set(row['predicted_label']), axis=1))
    print(f"top-n accuracy = {topn_pred/len(source_df)}")


def iou(row):
    true_x1 = row['new_x_1']
    true_y1 = row['new_y_1']
    true_x2 = row['new_x_2']
    true_y2 = row['new_y_2']
    pred_x1 = row['xmin']
    pred_y1 = row['ymin']
    pred_x2 = row['xmax']
    pred_y2 = row['ymax']
    inner_xmin = max(min(true_x1, true_x2), min(pred_x1, pred_x2))
    inner_xmax = min(max(true_x1, true_x2), max(pred_x1, pred_x2))
    inner_ymin = max(min(true_y1, true_y2), min(pred_y1, pred_y2))
    inner_ymax = min(max(true_y1, true_y2), max(pred_y1, pred_y2))
    # print(inner_xmin, inner_ymin, inner_xmax, inner_ymax)
    h1 = abs(true_y1 - true_y2)
    w1 = abs(true_x1 - true_x2)
    h2 = abs(pred_y1 - pred_y2)
    w2 = abs(pred_x1 - pred_x2)
    innerh = abs(inner_ymax - inner_ymin)
    innerw = abs(inner_xmax - inner_xmin)
    area_inner = innerh * innerw
    iou = (area_inner) / (h1 * w1 + h2 * w2 - area_inner)
    return (iou)


def get_parent_dir(n=1):
    """ returns the n-th parent dicrectory of the current
    working directory """
    current_path = os.path.dirname(os.path.abspath(__file__))
    for k in range(n):
        current_path = os.path.dirname(current_path)
    return current_path


src_path = os.path.join(get_parent_dir(0), 'src')
tiny_yolo_root = '/home/ilambda/goods_viewer/Debasish/tiny_yolo'
sys.path.append(src_path)

utils_path = os.path.join(get_parent_dir(1), 'Utils')
sys.path.append(utils_path)

import numpy as np
import tensorflow as tf

TF_VERSION2 = tf.__version__.startswith("2")
if TF_VERSION2: from tensorflow import keras
import keras.backend as K
from keras.layers import Input, Lambda
from keras.models import Model
from keras.optimizers import Adam
from keras.callbacks import TensorBoard, ModelCheckpoint, ReduceLROnPlateau, EarlyStopping
from keras_yolo3.yolo3.model import preprocess_true_boxes, yolo_body, tiny_yolo_body, yolo_loss
from keras_yolo3.yolo3.utils import get_random_data
from PIL import Image
from time import time
import pickle

from Train_Utils import get_classes, get_anchors, create_model, create_tiny_model, data_generator, \
    data_generator_wrapper, ChangeToOtherMachine

keras_path = os.path.join(src_path, 'keras_yolo3')
Data_Folder = os.path.join(get_parent_dir(1), 'Data')
# Image_Folder = os.path.join(Data_Folder,'Source_Images','Training_Images')
# VoTT_Folder = os.path.join(Image_Folder,'vott-csv-export')
YOLO_filename = os.path.join(tiny_yolo_root, 'training_labels.txt')

# Model_Folder = os.path.join(Data_Folder, 'Model_Weights')
YOLO_classname = os.path.join(tiny_yolo_root, 'category_names.txt')

log_dir = os.path.join(tiny_yolo_root, 'models')
anchors_path = os.path.join(tiny_yolo_root, 'yolo-tiny_anchors.txt')
weights_path = os.path.join(tiny_yolo_root, 'darknet-tiny-yolo.h5')

FLAGS = None

if __name__ == '__main__':
    # Delete all default flags
    parser = argparse.ArgumentParser(argument_default=argparse.SUPPRESS)
    '''
    Command line options
    '''

    parser.add_argument(
        "--annotation_file", type=str, default=YOLO_filename,
        help="Path to annotation file for Yolo. Default is " + YOLO_filename
    )
    parser.add_argument(
        "--classes_file", type=str, default=YOLO_classname,
        help="Path to YOLO classnames. Default is " + YOLO_classname
    )

    parser.add_argument(
        "--log_dir", type=str, default=log_dir,
        help="Folder to save training logs and trained weights to. Default is " + log_dir
    )

    parser.add_argument(
        "--anchors_path", type=str, default=anchors_path,
        help="Path to YOLO anchors. Default is " + anchors_path
    )

    parser.add_argument(
        "--weights_path", type=str, default=weights_path,
        help="Path to pre-trained YOLO weights. Default is " + weights_path
    )
    parser.add_argument(
        "--val_split", type=float, default=0.1,
        help="Percentage of training set to be used for validation. Default is 10 percent."
    )
    parser.add_argument(
        "--is_tiny", default=False, action="store_true",
        help="Use the tiny Yolo version for better performance and less accuracy. Default is False."
    )
    parser.add_argument(
        "--train_step1", default=False, action="store_true",
        help="Train the model with only the detection layer unfrozen. Default is False."
    )
    parser.add_argument(
        "--random_seed", type=int, default=43,
        help="Random seed value to make script deterministic. Default is 43, i.e. non-deterministic."
    )
    parser.add_argument(
        "--epochs", type=float, default=10,
        help="Number of epochs for training last layers and number of epochs for fine-tuning layers. Default is 10."
    )
    parser.add_argument(
        "--lr", type=float, default=0.0001,
        help="Learning rate for step_2: with all layers unfrozen. Default is 0.0001"
    )
    parser.add_argument(
        "--is_augment", default=False, action="store_true",
        help="Enable random train image augmentation Default is False."
    )
    parser.add_argument(
        "--infer", default=False, action="store_true", dest='inter',
        help="Invokes the detector script after the end of the training"
    )

    FLAGS = parser.parse_args()

    np.random.seed(FLAGS.random_seed)

    log_dir = FLAGS.log_dir
    if not os.path.exists(log_dir):
        os.mkdir(log_dir)

    class_names = get_classes(FLAGS.classes_file)
    num_classes = len(class_names)
    anchors = get_anchors(FLAGS.anchors_path)
    weights_path = FLAGS.weights_path

    input_shape = (448, 448)  # multiple of 32, height, width

    epoch1 = FLAGS.epochs if FLAGS.train_step1 else 0
    epoch2 = FLAGS.epochs

    is_tiny_version = (len(anchors) == 6)  # default setting
    if FLAGS.is_tiny:
        model = create_tiny_model(input_shape, anchors, num_classes,
                                  freeze_body=2, weights_path=weights_path)
    else:
        model = create_model(input_shape, anchors, num_classes,
                             freeze_body=2, weights_path=weights_path)  # make sure you know what you freeze

    log_dir_time = os.path.join(log_dir, '{}'.format(int(time())))
    logging = TensorBoard(log_dir=log_dir_time)
    checkpoint = ModelCheckpoint(os.path.join(log_dir, 'checkpoint.h5'),
                                 monitor='val_loss', save_weights_only=True, save_best_only=True, period=5)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=3, verbose=1)
    early_stopping = EarlyStopping(monitor='val_loss', min_delta=0, patience=10, verbose=1)

    val_split = FLAGS.val_split
    with open(FLAGS.annotation_file) as f:
        lines = f.readlines()

    # This step makes sure that the path names correspond to the local machine
    # This is important if annotation and training are done on different machines (e.g. training on AWS)
    # lines  = ChangeToOtherMachine(lines,remote_machine = '')
    np.random.shuffle(lines)
    num_val = int(len(lines) * val_split)
    num_train = len(lines) - num_val

    # Train with frozen layers first, to get a stable loss.
    # Adjust num epochs to your dataset. This step is enough to obtain a decent model.
    if FLAGS.train_step1:
        model.compile(optimizer=Adam(lr=1e-4), loss={
            # use custom yolo_loss Lambda layer.
            'yolo_loss': lambda y_true, y_pred: y_pred})

        batch_size = 32
        print('Train on {} samples, val on {} samples, with batch size {}.'.format(num_train, num_val, batch_size))
        history = model.fit_generator(
            data_generator_wrapper(lines[:num_train], batch_size, input_shape, anchors, num_classes,
                                   random=FLAGS.is_augment),
            steps_per_epoch=max(1, num_train // batch_size),
            validation_data=data_generator_wrapper(lines[num_train:], batch_size, input_shape, anchors, num_classes,
                                                   random=False),
            validation_steps=max(1, num_val // batch_size),
            epochs=epoch1,
            initial_epoch=0,
            callbacks=[logging, checkpoint])
        model.save_weights(os.path.join(log_dir, 'trained_weights_stage_1.h5'))

        step1_train_loss = history.history['loss']

        file = open(os.path.join(log_dir_time, 'step1_loss.npy'), "w")
        with open(os.path.join(log_dir_time, 'step1_loss.npy'), 'w') as f:
            for item in step1_train_loss:
                f.write("%s\n" % item)
        file.close()

        step1_val_loss = np.array(history.history['val_loss'])

        file = open(os.path.join(log_dir_time, 'step1_val_loss.npy'), "w")
        with open(os.path.join(log_dir_time, 'step1_val_loss.npy'), 'w') as f:
            for item in step1_val_loss:
                f.write("%s\n" % item)
        file.close()

    # Unfreeze and continue training, to fine-tune.
    # Train longer if the result is unsatisfactory.
    if True:
        for i in range(len(model.layers)):
            model.layers[i].trainable = True
        model.compile(optimizer=Adam(lr=FLAGS.lr),
                      loss={'yolo_loss': lambda y_true, y_pred: y_pred})  # recompile to apply the change
        print('Unfreeze all layers.')

        batch_size = 4  # note that more GPU memory is required after unfreezing the body
        print('Train on {} samples, val on {} samples, with batch size {}.'.format(num_train, num_val, batch_size))
        history = model.fit_generator(
            data_generator_wrapper(lines[:num_train], batch_size, input_shape, anchors, num_classes,
                                   random=FLAGS.is_augment),
            steps_per_epoch=max(1, num_train // batch_size),
            validation_data=data_generator_wrapper(lines[num_train:], batch_size, input_shape, anchors, num_classes,
                                                   random=False),
            validation_steps=max(1, num_val // batch_size),
            epochs=epoch1 + epoch2,
            initial_epoch=epoch1,
            callbacks=[logging, checkpoint, reduce_lr, early_stopping])
        model_path = os.path.join(log_dir, 'trained_weights_final.h5')
        model.save_weights(model_path)
        print(f"Model saved to : {model_path}")
        step2_train_loss = history.history['loss']

        file = open(os.path.join(log_dir_time, 'step2_loss.npy'), "w")
        with open(os.path.join(log_dir_time, 'step2_loss.npy'), 'w') as f:
            for item in step2_train_loss:
                f.write("%s\n" % item)
        file.close()

        step2_val_loss = np.array(history.history['val_loss'])

        file = open(os.path.join(log_dir_time, 'step2_val_loss.npy'), "w")
        with open(os.path.join(log_dir_time, 'step2_val_loss.npy'), 'w') as f:
            for item in step2_val_loss:
                f.write("%s\n" % item)
        file.close()

    if FLAGS.infer:
        output_dir = os.path.join(log_dir, "inference")
        out_df_path = os.path.join(output_dir, "Detection_Results.csv")
        source_df_path = os.path.join(tiny_yolo_root, "inference", "test_source_df2000.csv")
        if not os.path.exists(output_dir):
            os.mkdir(output_dir)
        command = f"python Detector.py --no_save_image --output {output_dir} --yolo_model {model_path} --classes {FLAGS.classes_file} --anchors {FLAGS.anchors_path}"
        subprocess.call(command, shell=True, cwd=os.path.join(get_parent_dir(1), "3_Inference"))
        out_df, source_df = create_pred_df(out_df_path, source_df_path, class_names)
        get_accuracies(out_df, source_df)


