import jieba
import tensorflow as tf


class DataHelper:
    
    def __init__(self, token_list, lower):
        self.token_list = token_list
        self.lower = lower
        self.char2id = {token: idx for idx, token in enumerate(token_list)}
    
    def get_seg_features(self, string):
        seg_feature = []
        for word in jieba.cut(string):
            if len(word) == 1:
                seg_feature.append(0)
            else:
                tmp = [2] * len(word)
                tmp[0] = 1
                tmp[-1] = 3
                seg_feature.extend(tmp)
        return seg_feature
    
    def get_sentence_features(self, sentence):
        def f(x):
            return x.lower() if self.lower else x
        chars = [self.char2id[f(w) if f(w) in self.token_list else '<UNK>'] for w in sentence]
        segs = self.get_seg_features("".join(sentence))
        return {"text": sentence, "char_ids": chars, "seg_ids": segs}


def file_based_convert_examples_to_features(
        examples, token_list, label_list, lower, output_file):
    writer = tf.python_io.TFRecordWriter(output_file)
    label2id = {label: idx for idx, label in enumerate(label_list)}
    d_helper = DataHelper(token_list, lower=lower)
    for (ex_index, example) in enumerate(examples):
        if ex_index % 10000 == 0:
            tf.logging.info("Writing example %d of %d" % (ex_index, len(examples)))
        
        features = d_helper.get_sentence_features(example.text)
        label_ids = [label2id[label] for label in example.label]
        
        tf_example = tf.train.Example(features=tf.train.Features(feature={
            "label_ids": tf.train.Feature(int64_list=tf.train.Int64List(value=label_ids)),
            "char_ids": tf.train.Feature(int64_list=tf.train.Int64List(value=features["char_ids"])),
            "seg_ids": tf.train.Feature(int64_list=tf.train.Int64List(value=features["seg_ids"]))
        }))
        writer.write(tf_example.SerializeToString())
    writer.close()

def file_based_input_fn_builder(input_file, is_training, batch_size, 
                                drop_remainder=False):
    """Creates an `input_fn` closure to be passed to Estimator."""
    name_to_features = {
        "char_ids": tf.VarLenFeature(tf.int64),
        "seg_ids": tf.VarLenFeature(tf.int64),
        "label_ids": tf.VarLenFeature(tf.int64),
    }
    
    def _decode_record(record, name_to_features):
        """Decodes a record to a TensorFlow example."""
        example = tf.parse_single_example(record, name_to_features)
        char_ids = tf.sparse_tensor_to_dense(example["char_ids"])
        seg_ids  = tf.sparse_tensor_to_dense(example["seg_ids"])
        label_ids = tf.sparse_tensor_to_dense(example["label_ids"])
        return char_ids, seg_ids, label_ids
    
    def input_fn():
        """The actual input function."""
        # For training, we want a lot of parallel reading and shuffling.
        # For eval, we want no shuffling and parallel reading doesn't matter.
        data = tf.data.TFRecordDataset(input_file)
        data = data.map(lambda record: _decode_record(record, name_to_features))
        if is_training:
            data = data.shuffle(buffer_size=10000)
        data = data.repeat(1).padded_batch(batch_size=batch_size, 
                          padded_shapes=([None], [None], [None])).prefetch(batch_size)
        iterator = data.make_one_shot_iterator()
        char_ids, seg_ids, label_ids = iterator.get_next()
        return {"char_ids": char_ids, "seg_ids": seg_ids}, label_ids
    
    return input_fn