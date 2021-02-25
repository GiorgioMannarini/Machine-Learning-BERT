import pkg_resources
import nltk
import re
import pandas as pd
import numpy as np

from nltk.corpus import stopwords
from nltk.corpus import wordnet
from nltk.stem import WordNetLemmatizer
from utility.emoticons_glove import EMOTICONS_GLOVE
from symspellpy import SymSpell

nltk.download('averaged_perceptron_tagger')
nltk.download('stopwords')
nltk.download('wordnet')


class Preprocessing:
  """
  Preprocesses the data and can even perform feature extraction.

  Attributes:
    __data: A pandas dataframe with the data (at least one column called text).
  """

  def __init__(self, list_: list, submission=False):
    """
    Builds the Pandas DataFrame.
      - If submission is False, a list of 2 elements is expected.
        The first one must be the negative tweets.
        The second one must be the positive tweets.
        The final DataFrame is composed of `text` and `label` columns.
      - If submission is True, a list of 1 element is expected.
        The final DataFrame is composed of `ids` and `text` columns.

    :param list_: a list of .txt files to be converted in DataFrame
    :type list_: list
    :param submission: specify the type of DataFrame (train or test data)
    :rtype submission: bool
    """

    if not submission:
      if len(list_) == 2:

        # Creating empty DataFrame
        self.__data = pd.DataFrame(columns=['text', 'label'])

        # Reading the content of each file in the list
        for i, file_name in enumerate(list_):
          with open(file_name) as f:
            content = f.read().splitlines()

          # Creating a DataFrame putting as label the position in the input list
          df = pd.DataFrame(columns=['text', 'label'],
                            data={'text': content,
                                  'label': np.ones(len(content)) * i})

          # Appending the dataframe
          self.__data = self.__data.append(df).reset_index(drop=True)

    else:
      if len(list_) == 1:
        # Reading the content
        with open(list_[0]) as f:
          content = f.read().splitlines()

        # Getting the ids
        ids = [line.split(',')[0] for line in content]
        # Getting the tweets' content
        texts = [','.join(line.split(',')[1:]) for line in content]

        # Creating the DataFrame
        self.__data = pd.DataFrame(columns=['ids', 'text'],
                                   data={'ids': ids, 'text': texts})

  # UTILITY METHODS

  def get(self):
    """
    Returns the DataFrame.

    :return: the DataFrame
    :rtype: pandas.DataFrame
    """
    return self.__data

  def logging(self):
    """
    Prints the first 10 rows in the dataframe stored in self.__data.
    """
    print('Logging:')
    print(self.__data['text'].head(10))

  def save_raw(self):
    """
    Creates a column in the dataframe as copy of `text` column
      to keep the original data.

    Must be called before anything else!
    """
    print('Saving raw tweet...')

    self.__data['raw'] = self.__data['text']

  # PREPROCESSING METHODS

  def drop_duplicates(self):
    """
    Removes duplicated in the dataframe according to text column.
    """
    print('Dropping duplicates...')

    self.__data = self.__data.drop_duplicates(subset=['text'])

  def remove_tags(self):
    """
    Removes tags (<user>, <url>) and final '...' characters (long tweets)
    """
    print('Removing tags...')

    self.__data['text'] = self.__data['text'].str.replace('<[\w]*>', '')
    self.__data['text'] = self.__data['text'].apply(lambda text: text.strip())
    self.__data['text'] = self.__data['text'].str.replace('\.{3}$', '')

  def convert_hashtags(self):
    """
    Removes '#' at the beginning of a tweet and corrects spacing of it.
    """
    print('Converting hashtags...')

    self.__data['text'] = self.__data['text'].str.replace(
      '(#)(\w+)',
      lambda text: Preprocessing.__word_segmentation(str(text.group(2))))

  def slangs_to_words(self):
    """
    Extends slangs to sequence of words.
    """
    print('Converting slangs to words...')

    # Reading the slangs from file
    with open('./utility/slang.txt') as f:
      chat_words_str = f.read().splitlines()

    # List of mappings {slang: slang_expanded}
    chat_words_map_dict = {}

    # List of slangs
    chat_words_list = []

    for line in chat_words_str:
      # Slang
      cw = line.split('=')[0]

      # Slang expanded
      cw_expanded = line.split('=')[1]

      # Appending slang and mapping
      chat_words_list.append(cw)
      chat_words_map_dict[cw] = cw_expanded

    # Make sure slangs in list are unique
    chat_words_list = set(chat_words_list)

    # Function to be called for each tweet
    def chat_words_conversion(text):
      new_text = []

      # For each word in the tweet
      for w in text.split():

        # If slangs is in the mapping
        if w.upper() in chat_words_list:
          new_text.append(chat_words_map_dict[w.upper()])

        # Otherwise, use the slang itself
        else:
          new_text.append(w)
      return ' '.join(new_text)

    # Calling `chat_words_conversion` for each tweet
    self.__data['text'] = self.__data['text'].apply(
      lambda text: chat_words_conversion(str(text)))

  def final_parenthesis(self):
    """
    Substitutes the final parenthesis of a tweet with a positive or negative smile.
    More on this in the report.
    """

    print('Substituting final paranthesis...')

    self.__data['text'] = self.__data['text'].str.replace('\)+$', ':)')
    self.__data['text'] = self.__data['text'].str.replace('\(+$', ':(')

  def final_parenthesis_to_tags(self):
    """
    Substitutes the final parenthesis of a tweet with a positive or negative smile tag.
    More on this in the report.
    """
    print('Substituting final paranthesis with tags...')

    self.__data['text'] = self.__data['text'].str.replace('\)+$',
                                                          ' <smile> ')
    self.__data['text'] = self.__data['text'].str.replace('\(+$',
                                                          ' <sadface> ')

  def remove_numbers(self):
    """
    Removes numbers from each tweet
    """

    print('Removing numbers...')
    self.__data['text'] = self.__data['text'].str.replace('\d', '')

  def remove_punctuation(self):
    """
    Removes everything that is not alphanumeric and not a space.
    """

    print('Removing punctuation...')
    self.__data['text'] = self.__data['text'].str.replace('[^\w\s]', '')

  def to_lower(self):
    """
    Converts each tweet to lowercase.
    """

    print('Converting to lowercase...')
    self.__data['text'] = self.__data['text'].str.lower()

  def correct_spelling(self):
    """
    Corrects spelling of each tweet.
    """

    print('Correcting spelling...')
    self.__data['text'] = self.__data['text'].apply(
      lambda text: Preprocessing.__correct_spelling(text))

  def lemmatize(self):
    """
    Performs the lemmatization.
    """

    print('Performing lemmatization...')
    self.__data['text'] = self.__data['text'].apply(Preprocessing.__lemmatize)

  def remove_stopwords(self):
    """
    Removes english stopwords.
    """

    print('Removing stopwords...')

    # Getting english stopwords set
    stopwords_ = set(stopwords.words('english'))

    # Removing stopwords for each tweet
    self.__data['text'] = self.__data['text'].apply(
      lambda text: ' '.join(
        [word for word in str(text).split() if word not in stopwords_]))

  def empty_tweets(self):
    """
    Adds tag <EMPTY> for empty tweets.
    """

    print('Marking empty tweets...')
    self.__data['text'] = self.__data['text'].str.replace('^\s*$', '<EMPTY>')

  def remove_elongs(self):
    """
    Removes elongs. (e.g.: hellooooo -> hello)
    """

    print('Removing elongs...')
    self.__data['text'] = self.__data['text'].apply(
      lambda text: str(re.sub(r'\b(\S*?)(.)\2{2,}\b', r'\1\2', text)))

  def correct_spacing_indexing(self):
    """
    Deletes double or more spaces and corrects indexing.

    Must be called after calling the above methods.
    Most of the above methods just delete a token. However since tokens are
    surrounded by whitespaces, they will often result in having more than one
    space between words.

    The only exception is for `remove_space_between_emoticons` method.
    Should be called before and after calling that method.
    It could exist ':  )' which that method doesn't recognize.
    """

    print('Correcting spacing...')

    # Removing double spaces
    self.__data['text'] = self.__data['text'].str.replace('\s{2,}', ' ')

    # Stripping text
    self.__data['text'] = self.__data['text'].apply(lambda text: text.strip())

    # Correcting the indexing
    self.__data.reset_index(inplace=True, drop=True)

  def remove_space_between_emoticons(self):
    """
    Removes spaces between emoticons (e.g.: ': )' --> ':)').
    Adds a space between a word and an emoticon (e.g.: 'hello:)' --> 'hello :)')
    """

    print('Removing space between emoticons...')

    # Getting list of all emoticons
    emo_list = [el for value in list(EMOTICONS_GLOVE.values()) for el in value]

    # Putting a space between each character in each emoticon
    emo_with_spaces = '|'.join(re.escape(' '.join(emo)) for emo in emo_list)

    # Getting all emoticons that don't contain any alphanumeric character
    all_non_alpha_emo = '|'.join(re.escape(emo) for emo in emo_list if not any(
      char.isalpha() or char.isdigit() for char in emo))

    # Removing spaces between emoticons
    self.__data['text'] = self.__data['text'].str.replace(
      emo_with_spaces,
      lambda t: t.group().replace(' ', ''))

    # Adding space between a word and an emoticon
    self.__data['text'] = self.__data['text'].str.replace(
      rf'({all_non_alpha_emo})',
      r' \1 ')

  def emoticons_to_tags(self):
    """
    Convert emoticons (with or without spaces) into tags
      according to the pretrained stanford glove model
      (e.g.: :) ---> <smile> and so on)
    """

    print('Converting emoticons to tags...')

    # Dictionary like {tag:[list_of_emoticons]}
    union_re = {}
    for tag, emo_list in EMOTICONS_GLOVE.items():
      # Getting emoticons as they are
      re_emo = '|'.join(re.escape(emo) for emo in emo_list)
      union_re[tag] = re_emo

    # Function to be called for each tweet
    def inner(text, _union_re):
      for tag, union_re in _union_re.items():
        text = re.sub(union_re, ' ' + tag + ' ', text)
      return text

    # Applying for each tweet
    self.__data['text'] = self.__data['text'].apply(
      lambda text: inner(str(text), union_re))

  def hashtags_to_tags(self):
    """
    Convert hashtags. (e.g.: #hello ---> <hashtag> hello)
    """

    print('Converting hashtags to tags...')
    self.__data['text'] = self.__data['text'].str.replace(r'#(\S+)',
                                                          r'<hashtag> \1')

  def numbers_to_tags(self):
    """
    Convert numbers into tags. (e.g.: 34 ---> <number>)
    """

    print('Converting numbers to tags...')
    self.__data['text'] = self.__data['text'].str.replace(
      r'[-+]?[.\d]*[\d]+[:,.\d]*', r'<number>')

  def repeat_to_tags(self):
    """
    Convert repetitions of '!' or '?' or '.' into tags.
      (e.g.: ... ---> . <repeat>)
    """

    print('Converting repetitions of symbols to tags...')
    self.__data['text'] = self.__data['text'].str.replace(r'([!?.]){2,}',
                                                          r'\1 <repeat>')

  def elongs_to_tags(self):
    """
    Convert elongs into tags. (e.g.: hellooooo ---> hello <elong>)
    """

    print('Converting elongated words to tags...')
    self.__data['text'] = self.__data['text'].str.replace(
      r'\b(\S*?)(.)\2{2,}\b', r'\1\2 <elong>')

  def remove_endings(self):
    """
    Remove ... <url> which represents the ending of tweet
    """

    print('Removing tweet ending when the tweet is cropped...')
    self.__data['text'] = self.__data['text'].str.replace(r'\.{3} <url>$', '')

  # STATIC METHODS (private, used internally)

  # Instance of `SymSpell` class
  symspell = None

  @staticmethod
  def __get_symspell():
    """
    Instantiates a `SymSpell` object.

    :return: instantiated object
    :rtype: SymSpell
    """

    # If is not already instantiated
    if Preprocessing.symspell is None:
      # Instantiating `SymSpell`
      Preprocessing.symspell = SymSpell()

      # Getting dictionary for single words
      dictionary_path = pkg_resources.resource_filename(
        'symspellpy',
        'frequency_dictionary_en_82_765.txt')
      Preprocessing.symspell.load_dictionary(dictionary_path, term_index=0,
                                             count_index=1)

      # Getting dictionary for bigram (two words)
      bigram_path = pkg_resources.resource_filename(
        'symspellpy',
        'frequency_bigramdictionary_en_243_342.txt')
      Preprocessing.symspell.load_bigram_dictionary(bigram_path, term_index=0,
                                                    count_index=2)

    return Preprocessing.symspell

  @staticmethod
  def __word_segmentation(text):
    """
    Tries to put spaces between word in a text (used for hashtag).
      (e.g.: helloguys --> hello guys))

    :param text: Text to be converted (typically an hashtag)
    :type text: str
    :return: Processed text
    :rtype: str
    """

    # `max_edit_distance = 0` avoids that `SymSpell` corrects spelling.
    result = Preprocessing.__get_symspell().word_segmentation(text,
                                                              max_edit_distance=0)
    return result.segmented_string

  @staticmethod
  def __correct_spelling(text):
    """
    Corrects spelling of a word (e.g.: helo -> hello)

    :param text: Text to be converted
    :type text: str
    :return: Processed text
    :rtype: str
    """

    # `max_edit_distance = 2` tells `SymSpell` to check at a maximum distance
    #  of 2 in the vocabulary. Only words with at most 2 letters wrong will be corrected.
    result = Preprocessing.__get_symspell().lookup_compound(text,
                                                            max_edit_distance=2)

    return result[0].term

  @staticmethod
  def __get_wordnet_tag(nltk_tag):
    """
    Returns type of word according to nltk pos tag.

    :param nltk_tag: nltk pos tag
    :type nltk_tag: list(tuple(str, str))
    :return: type of a word
    :rtype: str
    """

    if nltk_tag.startswith('V'):
      return wordnet.VERB
    elif nltk_tag.startswith('N'):
      return wordnet.NOUN
    elif nltk_tag.startswith('J'):
      return wordnet.ADJ
    elif nltk_tag.startswith('R'):
      return wordnet.ADV
    else:
      # This is the default in WordNetLemmatizer, when no pos tag is passed
      return wordnet.NOUN

  @staticmethod
  def __lemmatize(text):
    """
    Performs lemmatization using nltk pos tag and `WordNetLemmatizer`.

    :param text: Text to be processed
    :type text: str
    :return: processed texg
    :rtype: str
    """

    nltk_tagged = nltk.pos_tag(text.split())
    lemmatizer = WordNetLemmatizer()

    return ' '.join(
      [lemmatizer.lemmatize(w, Preprocessing.__get_wordnet_tag(nltk_tag))
       for w, nltk_tag in nltk_tagged])
