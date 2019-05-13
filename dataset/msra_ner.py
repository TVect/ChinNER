from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import codecs
import tarfile
import logging
from collections import namedtuple

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
        self._load_train_examples()
        self._load_test_examples()
        logging.info("train examples cnt: {}, test examples cnt: {}"
                     .format(len(self.train_examples), len(self.test_examples)))

    def _load_train_examples(self):
        train_file = os.path.join(self.dataset_dir, "msra_train_bio")
        self.train_examples = self._read_data(train_file)

    def _load_test_examples(self):
        self.test_file = os.path.join(self.dataset_dir, "msra_test_bio")
        self.test_examples = self._read_data(self.test_file)

    def get_train_examples(self):
        return self.train_examples

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

    def get_label_map(self):
        return self.label_map

    def _read_data(self, input_file):
        """read data"""
        NERItemClass = namedtuple('NERItemClass',['text', 'label'])
        examples = []
        example_dict = {"text": [], "label": []}
        with codecs.open(input_file, "r", encoding="UTF-8") as f:
            for line in f:
                if line.strip() == "":
                    if example_dict["text"] is not None:
                        examples.append(NERItemClass(**example_dict))
                    example_dict = {"text": [], "label": []}
                else:
                    toks = line.split()
                    if len(toks) == 2:
                        example_dict["text"].append(toks[0])
                        example_dict["label"].append(toks[1])
                    else:
                        example_dict["text"].append(" ")
                        example_dict["label"].append("O")
 
        return examples


if __name__ == "__main__":
    ds = MSRA_NER()
    for e in ds.get_train_examples()[:10]:
        print("{}\t{}".format(e.text, e.label))

