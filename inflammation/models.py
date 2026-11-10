"""Module containing models representing patients and their data.

The Model layer is responsible for the 'business logic' part of the software.

Patients' data is held in an inflammation table (2D array) where each row contains
inflammation data for a single patient taken over a number of days
and each column represents a single day across all patients.
"""

import numpy as np


def load_csv(filename):
    """Load a Numpy array from a CSV

    :param filename: Filename of CSV to load
    """
    return np.loadtxt(fname=filename, delimiter=",")


def daily_mean(data):
    """Calculate the daily mean of a 2D inflammation data array.

    :param data: 2D data inflammation array, with one row per patient
    :type data: numpy.Array (2D)
    :return: The mean inflammation for each patient
    :rtype: numpy.Array (1D)
    """
    return np.mean(data, axis=0)


def daily_max(data):
    """Calculate the daily max of a 2D inflammation data array.

    :param data: 2D data inflammation array, with one row per patient
    :type data: numpy.Array (2D)
    :return: The maximum inflammation for each patient
    :rtype: numpy.Array (1D)
    """
    return np.max(data, axis=0)


def daily_min(data):
    """Calculate the daily min of a 2D inflammation data array.

    :param data: 2D data inflammation array, with one row per patient
    :type data: numpy.Array (2D)
    :return: The minimum inflammation for each patient
    :rtype: numpy.Array (1D)
    """
    return np.min(data, axis=0)


def patient_normalise(data):
    """Normalise patient data from a 2D inflammation data array.

    NaN values are ignored, and normalised to 0.

    Negative values are rounded to 0.

    :param data: 2D data inflammation array, with one row per patient
    :type data: numpy.Array (2D)
    :raises ValueError: If any of the values are negative we cannot normalize.
    :raises TypeError: If the data are not an ndarray
    :return: The normalized inflammation for each patient
    :rtype: numpy.Array (1D)
    """
    if not isinstance(data, np.ndarray):
        raise TypeError('Inflammation values should be an np.ndarray')
    if np.any(data < 0):
        raise ValueError('Inflammation values should not be negative')

    max = np.nanmax(data, axis=1)
    with np.errstate(invalid='ignore', divide='ignore'):
        normalised = data / max[:, np.newaxis]
    normalised[np.isnan(normalised)] = 0
    normalised[normalised < 0] = 0
    return normalised