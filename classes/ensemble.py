import pandas as pd
import numpy as np


class Ensemble:

  def __init__(self, model_accuracies, model_names):
    """
    :param model_accuracies: a dictionary containing the path of all considered
              submissions as keys. The value is the accuracy of that model.
    :type model_accuracies: dict
    :param model_names: a list containing the names of the used models.
    :type model_names: list
    """

    self.__model_accuracies = model_accuracies
    self.__model_names = model_names

  def __build_dataframe(self):
    """
    This method builds a dataframe containing all the predictions of all the
      considered models, multiplied by the accuracy of the model. We use this
      dataframe to do a weighted voting among the models.
    """

    paths = list(self.__model_accuracies.keys())
    validation_accuracies = list(self.__model_accuracies.values())

    # Building a dataframe with the prediction of each model.
    # Explanation: we read each csv, using only the Prediction column.
    # We then multiply that column for the valitation accuracy of the related
    # model. We merge every csv in one single dataframe.
    ensemble_df = pd.concat([pd.read_csv(path, usecols=["Prediction"]).rename(
      columns={"Prediction": f"Prediction{self.__model_names[i]}"}) \
                             * validation_accuracies[i] for i, path in
                             enumerate(paths)], axis=1)

    # Updating the index to match with the submission one:
    ensemble_df.index += 1

    # Printing the result
    print(ensemble_df)

    return ensemble_df

  def predict(self, submission_path):
    """
    This method creates a submission file with the weighted voting in the model.
    """

    # Building the dataframe
    ensemble_df = self.__build_dataframe()

    # Predicting
    predictions = np.sign(ensemble_df.sum(axis=1))

    # Creating the submission file
    submission = pd.DataFrame(columns=['Id', 'Prediction'],
                              data={'Id': ensemble_df.index,
                                    'Prediction': predictions})

    submission.to_csv(submission_path, index=False)
