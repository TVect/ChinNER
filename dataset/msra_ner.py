from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import codecs
import tarfile
import logging
from collections import namedtuple
from sklearn.model_selection import train_test_split

DATA_HOME = os.path.abspath(os.path.dirname(__file__))

class MSRA_NER:
    """
    A set of manually annotated Chinese word-segmentation data and
    specifications for training and testing a Chinese word-segmentation system
    for research purposes.  For more information please refer to
    https://www.microsoft.com/en-us/download/details.aspx?id=52531
    """

    def __init__(self):
        self.dataset_dir = os.path.join(DATA_HOME, "msra")
        if not os.path.exists(self.dataset_dir):
            with tarfile.open(os.path.join(DATA_HOME, "msra.tar.gz"), "r:gz") as tar:
                tar.extractall()
        else:
            logging.info("Dataset {} already cached.".format(self.dataset_dir))

        self.train_examples, self.dev_examples = train_test_split(
            self._load_train_examples(), test_size=3000, random_state=17)
        self.test_examples = self._load_test_examples()
        logging.info("train examples cnt: {}, test examples cnt: {}"
                     .format(len(self.train_examples), len(self.test_examples)))

    def _load_train_examples(self):
        train_file = os.path.join(self.dataset_dir, "msra_train_bio")
        return self._read_data(train_file)

    def _load_test_examples(self):
        self.test_file = os.path.join(self.dataset_dir, "msra_test_bio")
        return self._read_data(self.test_file)

    def get_train_examples(self):
        return self.train_examples

    def get_dev_examples(self):
        return self.dev_examples

    def get_test_examples(self):
        return self.test_examples

    def get_labels(self):
        return ["B-PER", "I-PER", "B-ORG", "I-ORG", "B-LOC", "I-LOC", "O"]

    @property
    def num_labels(self):
        """
        Return the number of labels in the dataset.
        """
        return len(self.get_labels())

    def _read_data(self, input_file):
        ''' read data '''
        NERItemClass = namedtuple('NERItemClass',['text', 'label'])
        examples = []
        example_dict = {"text": [], "label": []}
        with codecs.open(input_file, 'r', 'utf8') as fr:
            for line in fr:
                line = line.rstrip()
                if not line:
                    if len(example_dict["text"]) > 0:
                        if 'DOCSTART' not in example_dict["text"][0]:
                            examples.append(NERItemClass(**example_dict))
                        example_dict = {"text": [], "label": []}
                else:
                    if line[0] == " ":
                        line = "$" + line[1:]
                        word = line.split()
                    elif len(line.split()) == 1:
                        word = ["$", 'O']
                    else:
                        word= line.split()
                    assert len(word) >= 2
                    example_dict["text"].append(word[0])
                    example_dict["label"].append(word[1])
            if len(example_dict["text"]) > 0:
                if 'DOCSTART' not in example_dict["text"][0]:
                    examples.append(NERItemClass(**example_dict))
        return examples


if __name__ == "__main__":
    ds = MSRA_NER()
    for e in ds.get_train_examples()[:10]:
        print("{}\t{}".format(e.text, e.label))

