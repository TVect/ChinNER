# -*- coding: utf-8 -*-

import collections
import tensorflow as tf

class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self, input_ids, input_mask, segment_ids, label_ids):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.label_ids = label_ids


def convert_single_example(ex_index, example, label_list, max_seq_length,
                           tokenizer):
    """Converts a single `InputExample` into a single `InputFeatures`."""
    tokens = example.text
    labels = example.label
    tag2id = {tag: id for id, tag in enumerate(label_list)}
    # 序列截断
    if len(tokens) >= max_seq_length - 1:
        tokens = tokens[0:(max_seq_length - 2)]  # -2 的原因是因为序列需要加一个句首和句尾标志
        labels = labels[0:(max_seq_length - 2)]

    ntokens = ["[CLS]"] # 句子开始设置CLS 标志
    segment_ids = [0, ]
    label_ids = [tag2id["O"]]

    for i, token in enumerate(tokens):
        ntokens.extend(tokenizer.tokenize(token))
        segment_ids.append(0)
        label_ids.append(tag2id[labels[i]])
    ntokens.append("[SEP]")
    segment_ids.append(0)
    label_ids.append(tag2id["O"])   # 句尾添加[SEP] 标志
    input_ids = tokenizer.convert_tokens_to_ids(ntokens)  # 将序列中的字(ntokens)转化为ID形式
    input_mask = [1] * len(input_ids)
    
    # padding
    while len(input_ids) < max_seq_length:
        input_ids.append(0)
        input_mask.append(0)
        segment_ids.append(0)
        # we don't concerned about it!
        label_ids.append(0)
        ntokens.append("**NULL**")
        # label_mask.append(0)
    # print(len(input_ids))
    assert len(input_ids) == max_seq_length
    assert len(input_mask) == max_seq_length
    assert len(segment_ids) == max_seq_length
    assert len(label_ids) == max_seq_length
    # assert len(label_mask) == max_seq_length

    # 结构化为一个类
    feature = InputFeatures(input_ids=input_ids, 
                            input_mask=input_mask, 
                            segment_ids=segment_ids,
                            label_ids=label_ids)
    return feature
    
def file_based_convert_examples_to_features(
    examples, label_list, max_seq_length, tokenizer, output_file):
    """Convert a set of `InputExample`s to a TFRecord file."""
    
    writer = tf.python_io.TFRecordWriter(output_file)
    
    for (ex_index, example) in enumerate(examples):
        if ex_index % 10000 == 0:
            tf.logging.info("Writing example %d of %d" % (ex_index, len(examples)))

        features = collections.OrderedDict()
        feature = convert_single_example(ex_index, example, label_list, max_seq_length, tokenizer)
        features["input_ids"] = tf.train.Feature(
            int64_list=tf.train.Int64List(value=feature.input_ids))
        features["input_mask"] = tf.train.Feature(
            int64_list=tf.train.Int64List(value=feature.input_mask))
        features["segment_ids"] = tf.train.Feature(
            int64_list=tf.train.Int64List(value=feature.segment_ids))
        features["label_ids"] = tf.train.Feature(
            int64_list=tf.train.Int64List(value=feature.label_ids))
        tf_example = tf.train.Example(features=tf.train.Features(feature=features))
        
        writer.write(tf_example.SerializeToString())
    writer.close()

def file_based_input_fn_builder(input_file, seq_length, is_training,
                                drop_remainder):
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
            d = d.repeat()
            d = d.shuffle(buffer_size=100)

        d = d.apply(tf.contrib.data.map_and_batch(
            lambda record: _decode_record(record, name_to_features),
            batch_size=batch_size,
            drop_remainder=drop_remainder))
        return d
    
    return input_fn
