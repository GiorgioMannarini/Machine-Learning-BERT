import pandas as pd
from sklearn.model_selection import train_test_split
from abc import ABC, abstractmethod
from classes.preprocessing import Preprocessing
from constants import *


class AbstractModel(ABC):
  """
  Abstract class that models a generic neural network. Will be extended
    by GRU, BERT and Baseline
  """

  def __init__(self, weights_path):
    """
    :param weights_path: weights path of the model. Model's parameters will
            be loaded and saved from this path.
    :type weights_path: str
    """

    self._weights_path = weights_path

  @abstractmethod
  def get_preprocessing_methods(self, istest=False):
    """
    Specifies the Preprocessing class methods to be called for BERT.

    :param istest: Specify whether methods to be called are for test data
    :type istest: bool
    :return: A list of all sorted methods to be called
    :rtype: list
    """

    pass

  @abstractmethod
  def fit_predict(self, X, Y, ids_test, X_test, prediction_path):
    """
    Abstract method. Its implementation will fit (train) the model, and makes
      prediction on the test data.

    :param X: datapoint matrix. Will be splitted into training and validation data.
    :type X: numpy.ndarray
    :param Y: labels of the datapoints.
    :type Y: numpy.ndarray
    :param ids_test: the ids of the test datapoints, necessary to make a prediction.
    :type ids_test: numpy.ndarray
    :param X_test: the matrix containing the test datapoints for the prediction.
    :type X_test: numpy.ndarray
    :param prediction_path: relative path of the prediction file.
    :type prediction_path: str
    """

    pass

  @abstractmethod
  def predict(self, ids, X, path):
    """
    Abstract method. Its implementation will perform the predictions.
      Usually called within the fit_predict method.

    :param ids: ids of testing data.
    :type ids: numpy.ndarray
    :param X: matrix of the testing datapoints.
    :type x: numpy.ndarray
    :param path: specifies where to store the submission file
    :type path: str
    """

    pass

  @staticmethod
  def _create_submission(ids, predictions, path):
    """
    Static method used to generate the submission file after a prediction.

    :param ids: ids of testing data.
    :type ids: numpy.ndarray
    :param predictions: array of the predicted labels. Each element can be 1 or -1.
    :type predictions: numpy.ndarray
    :param path: specifies where to store the submission file
    :type path: str
    """

    # Generating the submission file
    submission = pd.DataFrame(columns=['Id', 'Prediction'],
                              data={'Id': ids, 'Prediction': predictions})

    # For many models the labels are 0 or 1. Replacing 0s with -1s.
    submission['Prediction'].replace(0, -1, inplace=True)

    # Saving the file
    submission.to_csv(path, index=False)

  @staticmethod
  def _split_data(X, Y, split_size=0.2):
    """
    Static method used to split the datapoints into train and validation set.

    :param X: datapoint matrix. Will be splitted into training and validation data.
    :type X: numpy.ndarray
    :param Y: labels of the datapoints.
    :type Y: numpy.ndarray
    :param split_size: size of the testing data with respecto to the training set.
    :type split_size: float,  optional.
    :return: the training and validation matrices with their respective labels.
    :rtype: tuple
    """

    print('Splitting data in train and test set...')
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y,
                                                        test_size=split_size)

    return X_train, X_test, Y_train, Y_test
