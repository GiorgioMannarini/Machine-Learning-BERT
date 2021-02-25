from classes.abstract_model import AbstractModel
from transformers import BertTokenizer, TFBertForSequenceClassification
from transformers import InputExample, InputFeatures, AdamWeightDecay, WarmUp
import tensorflow as tf


class Bert(AbstractModel):
  """
  This class implements a Bert model, pretrained, provided by Hugging Face
    and developed by Google.
  """

  def __init__(self, weights_path):
    """
    :param weights_path: specifies where load/store weights of the model
    :type weights_path: str
    """
    super().__init__(weights_path)

    # A tensorflow model of Bert base (uncased), pre-trained.
    # More on it on our report.
    self.__model = TFBertForSequenceClassification.from_pretrained(
      'bert-large-uncased')

    # Instanciating a proper tokenizer for Bert
    self.__tokenizer = BertTokenizer.from_pretrained('bert-large-uncased')

  def get_preprocessing_methods(self, istest=False):
    """
    Specifies the Preprocessing class methods to be called for BERT.

    :param istest: Specify whether methods to be called are for test data
    :type istest: bool
    :return: A list of all sorted methods to be called
    :rtype: list
    """

    methods = []

    if not istest:
      methods.append("drop_duplicates")

    methods.extend([
      'to_lower',
      'remove_tags',
      'final_parenthesis',
      'correct_spacing_indexing',
      'remove_space_between_emoticons',
      'correct_spacing_indexing'
    ])

    return methods

  def fit_predict(self, X, Y, ids_test, X_test, prediction_path, batch_size=24,
                  epochs=3):
    """
    Fits (train) the model, and makes a prediction on the test data.

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
    :param batch_size: size of the mini-batches used when training the model.
    :type batch_size: int, optional
    :param epochs: number of epochs used when training the model.
    :type epochs: int, optional
    """

    # Converting the tweets to have a good input for BERT.
    # Bert works indeed with input examples that have a specific structure.
    train_input_examples, validation_input_examples = \
      self.__convert_data_to_examples(X=X, Y=Y, split_size=0.1)

    train_data_size = len(train_input_examples)

    # Converting the previously obtained InputExamples in a
    # TensorFlow dataset, in order to train the model.
    train_data = self.__convert_examples_to_tf_dataset(
      list(train_input_examples))

    # Shuffling the data and combining consecutive elements of the dataset
    # into batches
    train_data = train_data.shuffle(100).batch(batch_size)

    # Same for validation data
    validation_data = self.__convert_examples_to_tf_dataset(
      list(validation_input_examples))
    validation_data = validation_data.batch(batch_size)

    # Computing the number of step per epoch
    steps_per_epoch = int(train_data_size / batch_size)

    # Computing the total number of training steps
    num_train_steps = steps_per_epoch * epochs

    print(f'Number of train steps: {num_train_steps}')

    # Setting a variable learning rate. In particular, after the 
    # warmup the learning rate will decrease linearly with the number of steps.
    decay_schedule = tf.keras.optimizers.schedules.PolynomialDecay(
      initial_learning_rate=2e-5,
      decay_steps=num_train_steps,
      end_learning_rate=0)

    # Setting a liear warmup to the model's learning rate. The lr will start
    # from 0 and will end at initial_learing_rate, increasing linearly.
    # The number of warmup steps is 10% of the number of total steps.
    warmup_schedule = WarmUp(
      initial_learning_rate=2e-5,
      decay_schedule_fn=decay_schedule,
      warmup_steps=(num_train_steps * 0.1))

    # Defining the optimizer. Gradient norm clipped at 1.
    # NOTE: we used AdamWeightDecay from transformers, and not Adam from
    # tensorflow, to set the warmup_schedule, but the weight decay is 0.
    optimizer = AdamWeightDecay(learning_rate=warmup_schedule,
                                epsilon=1e-08,
                                clipnorm=1.0)

    # Defining the loss and the evaluation metric.
    loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    metric = tf.keras.metrics.SparseCategoricalAccuracy('accuracy')

    # Compiling the model
    self.__model.compile(optimizer=optimizer, loss=loss, metrics=[metric])

    # Fitting the model
    print('Fitting the model...')
    self.__model.fit(train_data, epochs=epochs, validation_data=validation_data)

    # Saving model's weights
    print('Saving the weights...')
    self.__model.save_pretrained(f'{self._weights_path}model')

    # Calling the predict method to make predictions
    print('Predicting...')
    self.predict(ids_test, X_test, prediction_path, from_weights=False)

  def predict(self, ids, X, path, from_weights=True):
    """
    Performs the predictions. Usually called within the fit_predict method.

    :param ids: ids of testing data.
    :type ids: numpy.ndarray
    :param X: matrix of the testing datapoints.
    :type x: numpy.ndarray
    :param path: specifies where to store the submission file
    :type path: str
    :param from_weights: specifies if it is a prediction of a new model or if
            it is made according to a pre-trained one.
    :type from_weights: bool, optional
    """

    # If we are making a prediction on a pre-trained model, we load it here.
    if from_weights:
      self.__model = TFBertForSequenceClassification.from_pretrained(
        f'{self._weights_path}model')

    predictions = []

    # The classical tensorflow  model.predict() doesn't work with bert.
    # We created out prediction pipeline.
    for i, tweet in enumerate(X):
      feature = self.__tokenizer.encode_plus(text=tweet, return_tensors='tf')

      # For each test datapoint bert returns a 2-element array with softmax values.
      # We take the argmax of that array (0 or 1) as predicted label.
      output = self.__model(feature)[0].numpy().squeeze().argmax()
      predictions.append(output)

      # Used in logging. We print the number of predicted tweets.
      if i % 100 == 0:
        print(f'Step: {i}')

    # Creating the submission file
    AbstractModel._create_submission(ids, predictions, path)

  def __convert_examples_to_tf_dataset(self, data, max_length=128):
    """
    Performs the tokenization where each word of each document has a max_length
      and returns a tensorflow dataset.
    Every element of the dataset consists of:
      1. a dict with the tokenized text and the attention mask, used
          to specify which tokens are valid and which ones are used for padding
      2. the tweet label

    This format is known and used by Bert

    :param data: input data. A list of InputExample objects.
    :type data: list
    :param max_length: fixed length of the tokenization
    :type max_length: int, optiona
    :return: a tensorflow dataset as described before
    :rtype: tf.data.Dataset
    """

    # A list of InputFeatures of a single tweet. Every feature contains:
    # tweet's tokens, tweet's attention mask, tweet's label.
    # For more info: https://huggingface.co/transformers/main_classes/processors.html#transformers.data.processors.utils.InputFeatures
    features = []

    for sample in data:
      # For every tweet creates a dictionary. This dictionary contains tweet's
      # tokens ('input_ids') and the tweet's attention mask ('attention mask').
      input_dict = self.__tokenizer(
        # The tweet itself. Remember that the sample is an InputExample.
        sample.text_a,
        # Specify to add the padding
        add_special_tokens=True,
        # Fixed tweet vector length
        max_length=max_length,
        # Not needed because we are not comparing text_a to text_b,
        # since we don't have a text_b.
        # For more info: https://huggingface.co/transformers/glossary.html#token-type-ids
        return_token_type_ids=False,
        # Specify to return a binary vector of lenght = max_length. The vector
        # takes 1 when the corresponding token in the tweet representation is
        # valid, 0 if it is a special character used for padding.
        # For more info: https://huggingface.co/transformers/glossary.html#attention-mask
        return_attention_mask=True,
        # Padding added to the right
        padding='max_length',
        # Truncate the tweet if it is longer than 128 words
        truncation=True
      )

      input_ids, attention_mask = (
        input_dict['input_ids'],
        input_dict['attention_mask'])

      # For every tweet it creates an object of type InputFeatures
      # and adds it to the list.
      features.append(
        InputFeatures(
          input_ids=input_ids, attention_mask=attention_mask,
          label=sample.label
        )
      )

    # Creating a generator to convert the features list into a tensorflow dataset.
    def gen():
      for f in features:
        yield (
          {
            'input_ids': f.input_ids,
            'attention_mask': f.attention_mask,
          },
          f.label,
        )

    # Returns the dataset from the generator.
    # For more info: https://www.tensorflow.org/api_docs/python/tf/data/Dataset#from_generator
    return tf.data.Dataset.from_generator(
      gen,
      (
        {
          'input_ids': tf.int32,
          'attention_mask': tf.int32,
        },
        tf.int64
      ),
      (
        {
          'input_ids': tf.TensorShape([None]),
          'attention_mask': tf.TensorShape([None]),
        },
        tf.TensorShape([]),
      ),
    )

  @staticmethod
  def __convert_data_to_examples(X, Y, split_size=0.2):
    """
    Function to transform the data in a format suitable for BERT.
    To know more about this format please refer to our report.

    :param X: input data that has to be converted to InputExamples
    :type X: numpy.ndarray
    :param Y: input labels
    :type X: numpy.ndarray
    :param split_size: specifies the ratio to split data in train/test
    :type split_size: float, optional
    :return: transformed data
    :rtype: tuple
    """

    # Splitting data in train and validation sets
    X_train, X_test, Y_train, Y_test = AbstractModel._split_data(
      X=X,
      Y=Y,
      split_size=split_size)

    # Generating input examples from each tweet
    train_input_examples = []
    for text, label in zip(X_train, Y_train):
      train_input_examples.append(
        InputExample(guid=None, text_a=text, text_b=None, label=label))

    validation_input_examples = []
    for text, label in zip(X_test, Y_test):
      validation_input_examples.append(
        InputExample(guid=None, text_a=text, text_b=None, label=label))

    return train_input_examples, validation_input_examples
