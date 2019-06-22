# -*- coding: utf-8 -*-

import json
import collections
import tensorflow as tf
from .bert import tokenization


class DataProcessor:

    def __init__(self, vocab_file, do_lower_case, label_list, max_seq_length=128):
        self.vocab_file = vocab_file
        self.do_lower_case = do_lower_case
        self.label_list = label_list
        self.label2id = {label: idx for idx, label in enumerate(label_list)}
        self.max_seq_length = max_seq_length
        self.tokenizer = tokenization.FullTokenizer(vocab_file=vocab_file, 
                                                    do_lower_case=do_lower_case)
    
    def dump_to_file(self, config_file):
        config = {"vocab_file": self.vocab_file, 
                  "do_lower_case": self.do_lower_case, 
                  "label_list": self.label_list,
                  "max_seq_length": self.max_seq_length}
        with open(config_file, "w") as fp:
            json.dump(config, fp, indent=4)

    @classmethod
    def load_from_file(cls, config_file):
        with open(config_file) as fr:
            config = json.load(fr)
        return cls(**config)
        
    def convert_tokens_to_ids(self, tokens):
        return self.tokenizer.convert_tokens_to_ids(tokens)
    
    def convert_labels_to_ids(self, labels):
        return [self.label2id.get(label) for label in labels]
    
    def convert_ids_to_labels(self, label_ids):
        return [self.label_list[label_id] for label_id in label_ids]
    
    def convert_example_to_features(self, example):
        """Converts a `NERItemClass` Example into a single `InputFeatures`."""
        tokens = example.text
        labels = example.label
        
        if labels == None:
            labels = ["O" for _ in range(len(tokens))]

        if len(tokens) >= self.max_seq_length - 1:
            tokens = tokens[0:(self.max_seq_length - 2)]  # we will add token: [CLS], [SEP]
            labels = labels[0:(self.max_seq_length - 2)]
        input_mask = [1] * (len(tokens) + 2) + [0] * (self.max_seq_length - len(tokens) - 2)
        tokens = ["[CLS]"] + [self.tokenizer.tokenize(token)[0] for token in tokens] + \
                 ["[SEP]"] + ["[PAD]" for _ in range(self.max_seq_length-len(tokens)-2)]
        labels = ["O"] + labels + ["O" for _ in range(self.max_seq_length-len(labels)-1)]
        segment_ids = [0] * self.max_seq_length
        
        input_ids = self.convert_tokens_to_ids(tokens)
        label_ids = self.convert_labels_to_ids(labels)

        assert len(input_ids) == self.max_seq_length
        assert len(input_mask) == self.max_seq_length
        assert len(segment_ids) == self.max_seq_length
        assert len(label_ids) == self.max_seq_length, "label_ids:{}, max_length:{}".format(len(label_ids), self.max_seq_length)

        features = collections.OrderedDict()
        features["input_ids"] = tf.train.Feature(
            int64_list=tf.train.Int64List(value=input_ids))
        features["input_mask"] = tf.train.Feature(
            int64_list=tf.train.Int64List(value=input_mask))
        features["segment_ids"] = tf.train.Feature(
            int64_list=tf.train.Int64List(value=segment_ids))
        features["label_ids"] = tf.train.Feature(
            int64_list=tf.train.Int64List(value=label_ids))
        tf_example = tf.train.Example(features=tf.train.Features(feature=features))
    
        return tf_example

    
def file_based_convert_examples_to_features(examples, data_processor, output_file):
    """Convert a set of `NERItemClass` examples to a TFRecord file."""
    
    writer = tf.python_io.TFRecordWriter(output_file)
    
    for (ex_index, example) in enumerate(examples):
        if ex_index % 10000 == 0:
            tf.logging.info("Writing example %d of %d" % (ex_index, len(examples)))
        tf_example = data_processor.convert_example_to_features(example)
        writer.write(tf_example.SerializeToString())
    writer.close()


def file_based_input_fn_builder(input_file, seq_length, is_training):
    """Creates an `input_fn` closure to be passed to Estimator."""
    name_to_features = {
        "input_ids": tf.FixedLenFeature([seq_length], tf.int64),
        "input_mask": tf.FixedLenFeature([seq_length], tf.int64),
        "segment_ids": tf.FixedLenFeature([seq_length], tf.int64),
        "label_ids": tf.FixedLenFeature([seq_length], tf.int64),
    }
    
    def _decode_record(record, name_to_features):
        """Decodes a record to a TensorFlow example."""
        example = tf.parse_single_example(record, name_to_features)
        # tf.Example only supports tf.int64, but the TPU only supports tf.int32.
        # So cast all int64 to int32.
        for name in list(example.keys()):
            t = example[name]
            if t.dtype == tf.int64:
                t = tf.to_int32(t)
            example[name] = t
        return example

    def input_fn(params):
        """The actual input function."""
        batch_size = params["batch_size"]
        # For training, we want a lot of parallel reading and shuffling.
        # For eval, we want no shuffling and parallel reading doesn't matter.
        d = tf.data.TFRecordDataset(input_file)
        if is_training:
            # d = d.repeat()
            d = d.shuffle(buffer_size=100)

        d = d.apply(tf.contrib.data.map_and_batch(
            lambda record: _decode_record(record, name_to_features),
            batch_size=batch_size))
        return d
    
    return input_fn


def serving_fn_builder(seq_length):
    feature_spec = {
            "input_ids": tf.FixedLenFeature([seq_length], tf.int64),
            "input_mask": tf.FixedLenFeature([seq_length], tf.int64),
            "segment_ids": tf.FixedLenFeature([seq_length], tf.int64),
            "label_ids": tf.FixedLenFeature([seq_length], tf.int64)
        }
    serving_fn = tf.estimator.export.build_parsing_serving_input_receiver_fn(feature_spec)
    return serving_fn
