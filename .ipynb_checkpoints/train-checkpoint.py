"""
Created on Tue Oct 12 10:56:09 2021

@author: marrog

Main file for training ViSual_IceD
"""
import os
import numpy as np
from keras import backend as K
from keras import callbacks
from src.model.visualiced_model import visual_iced
from src.utils.data_parser import DataParser

# Manually define hyper-parameters.
BATCH_SIZE = 2
EPOCHS = 50
LEARNING_RATE = 0.005
CLASSES = 2
UPSAMPLE_METHOD = 'gaussian'
START_NEURONS = 16

# File patch containing all training patches.
TRAINING_IMAGE_FP = ""
# Text file containing all list of all patch filenames.
TRAINING_IMAGE_LIST = ""


def generate_minibatches(data_parser, train=True):
    """
    Parameters
    ----------
    data_parser : class object
        DESCRIPTION. Class for extracting training and validation image patches
        to use within that epoch.
    train : TYPE, boolean
        DESCRIPTION. Determines whether the model is training
        or validating. The default is True.

    Returns
    -------
    None.

    """
    while True:
        if train:
            batch_ids = np.random.choice(data_parser.training_ids,
                                         data_parser.batch_size_train)
        else:
            batch_ids = np.random.choice(data_parser.validation_ids,
                                         data_parser.batch_size_train*2)
        im_vis, im_sar, ems, _ = data_parser.get_batch(batch_ids)
        yield([im_vis, im_sar], ems)


if __name__ == "__main__":
    # Define filename for model, weights and csv containing validation metrics
    # per epoch and directory to save files.
    MODEL_NAME = '.json'
    MODEL_DIR = os.path.join('', MODEL_NAME)
    CSV_FN = os.path.join(MODEL_DIR, '.csv')
    CHECKPOINT_FN = os.path.join(MODEL_DIR, '.hdf5')

    # define that images are in format (height, width, band).
    K.set_image_data_format('channels_last')
    K.image_data_format()
    
    # Create object of DataParser class which extracts the training and
    # validation patches used in each epoch.
    data_parser = DataParser(BATCH_SIZE, TRAINING_IMAGE_FP,
                            TRAINING_IMAGE_LIST)

    # Make object of CNN model
    model = visual_iced(CLASSES, UPSAMPLE_METHOD, LEARNING_RATE)

    # Define callbacks: ModelCheckpoint- save validation metric information.
    # csv_logger- Save validation metrics per epoch in csv.
    # tensorboard- save info as Tensorboard object.
    checkpointer = callbacks.ModelCheckpoint(filepath=CHECKPOINT_FN, verbose=0,
                                             monitor='val_loss',
                                             save_best_only=True)
    csv_logger = callbacks.CSVLogger(CSV_FN, append=True, separator=';')
    tensorboard = callbacks.TensorBoard(log_dir=MODEL_DIR, histogram_freq=0,
                                        write_graph=False, write_grads=False,
                                        write_images=False)

    # Train model for defined number of epochs.
    TRAIN_HISTORY = model.fit(generate_minibatches(data_parser,),
                              workers=1, epochs=EPOCHS,
                              steps_per_epoch=data_parser.steps_per_epoch,
                              validation_data=generate_minibatches(data_parser,
                                                             train=False),
                              validation_steps=data_parser.validation_steps,
                              callbacks=[checkpointer, csv_logger, tensorboard])

    # Save model and weights using manually defined filenames.
    model_json = model.to_json()
    json_fp = os.path.join(MODEL_DIR, ".json")
    with open(json_fp, "w") as JSON_FILE:
        JSON_FILE.write(model_json)
    weights_fn = os.path.join(MODEL_DIR, '.hdf5')
    model.save_weights(weights_fn)
