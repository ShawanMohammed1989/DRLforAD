"""
Master Thesis project by Artur Niederfahrenhorst
This file defines our data preprocessing.
"""

import numpy as np


def preprocess_images(image, configuration):
    """
    Depending on the configuration, crop, grayscale and shift image mean.
    :param image: RBG image
    :param configuration: configuration object
    :return: preprocessed image
    """
    if configuration['CROPIMAGE']:
        image = image[:, configuration['FROM_Y']:configuration['TO_Y'], configuration['FROM_X']:configuration['TO_X'], :]
    # we crop side of screen as they carry little information
    else:
        image = image
    if configuration['MEANSHIFT']:
        image = np.subtract(image, 127)
        image = np.divide(image, 127.0)
    else:
        image = np.divide(image, 255.0)

    return image
