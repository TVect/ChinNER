import os
import collections
import tensorflow as tf
from .bert import tokenization, modeling, optimization


tf.enable_eager_execution()
tf.logging.set_verbosity(tf.logging.INFO)

flags = tf.flags

FLAGS = flags.FLAGS

FILE_HOME = os.path.abspath(os.path.dirname(__file__))

## Required parameters
flags.DEFINE_string(
    "data_dir", os.path.join(FILE_HOME, "../../dataset/msra"), 
    "The input data dir")
flags.DEFINE_string(
    "bert_config_file", 
    os.path.join(FILE_HOME, "./bert_models/chinese_L-12_H-768_A-12/bert_config.json"), 
    "The config json file corresponding to the pre-trained BERT model. "
    "This specifies the model architecture.")
flags.DEFINE_string(
    "vocab_file", 
    os.path.join(FILE_HOME, "./bert_models/chinese_L-12_H-768_A-12/vocab.txt"),
    "The vocabulary file that the BERT model was trained on.")
flags.DEFINE_string(
    "output_dir", os.path.join(FILE_HOME, "./output"),
    "The output directory where the model checkpoints will be written.")

## Other parameters
flags.DEFINE_string(
    "init_checkpoint", os.path.join(FILE_HOME, "./bert_models/chinese_L-12_H-768_A-12/bert_model.ckpt"),
    "Initial checkpoint (usually from a pre-trained BERT model).")
flags.DEFINE_bool(
    "do_lower_case", True,
    "Whether to lower case the input text. Should be True for uncased "
    "models and False for cased models.")

flags.DEFINE_integer(
    "max_seq_length", 128,
    "The maximum total input sequence length after WordPiece tokenization. "
    "Sequences longer than this will be truncated, and sequences shorter "
    "than this will be padded.")

flags.DEFINE_bool("do_train", True, "Whether to run training.")
flags.DEFINE_bool("do_eval", False, "Whether to run eval on the dev set.")
flags.DEFINE_bool("do_predict", False, "Whether to run in inference mode on the test set.")
flags.DEFINE_integer("train_batch_size", 32, "Total batch size for training.")
flags.DEFINE_integer("eval_batch_size", 8, "Total batch size for eval.")
flags.DEFINE_integer("predict_batch_size", 8, "Total batch size for predict.")
flags.DEFINE_float("learning_rate", 5e-5, "The initial learning rate for Adam.")
flags.DEFINE_float("num_train_epochs", 3.0,
                   "Total number of training epochs to perform.")
flags.DEFINE_float(
    "warmup_proportion", 0.1,
    "Proportion of training to perform linear learning rate warmup for. "
    "E.g., 0.1 = 10% of training.")

flags.DEFINE_integer("save_checkpoints_steps", 1000,
                     "How often to save the model checkpoint.")

flags.DEFINE_integer("iterations_per_loop", 1000,
                     "How many steps to make in each estimator call.")

flags.DEFINE_bool("use_tpu", False, "Whether to use TPU or GPU/CPU.")

tf.flags.DEFINE_string(
    "tpu_name", None,
    "The Cloud TPU to use for training. This should be either the name "
    "used when creating the Cloud TPU, or a grpc://ip.address.of.tpu:8470 "
    "url.")

tf.flags.DEFINE_string(
    "tpu_zone", None,
    "[Optional] GCE zone where the Cloud TPU is located in. If not "
    "specified, we will attempt to automatically detect the GCE project from "
    "metadata.")

tf.flags.DEFINE_string(
    "gcp_project", None,
    "[Optional] Project name for the Cloud TPU-enabled project. If not "
    "specified, we will attempt to automatically detect the GCE project from "
    "metadata.")

tf.flags.DEFINE_string("master", None, "[Optional] TensorFlow master URL.")

flags.DEFINE_integer(
    "num_tpu_cores", 8,
    "Only used if `use_tpu` is True. Total number of TPU cores to use.")

class InputExample(object):
    """A single training/test example for simple sequence classification."""

    def __init__(self, guid=None, text=None, label=None):
        """Constructs a InputExample.
        Args:
          guid: Unique id for the example.
          text_a: list of word. The untokenized text of the first sequence. For 
            single sequence tasks, only this sequence must be specified.
          label: (Optional) list of tag. The label of the example. This should be
            specified for train and dev examples, but not for test examples.
        """
        self.guid = guid
        self.text = text
        self.label = label

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

def create_model(bert_config, is_training, input_ids, input_mask,
                 segment_ids, labels, num_labels, use_one_hot_embeddings):
    model = modeling.BertModel(
        config=bert_config,
        is_training=is_training,
        input_ids=input_ids,
        input_mask=input_mask,
        token_type_ids=segment_ids)

    output_layer = model.get_sequence_output()
    hidden_size = output_layer.shape[-1].value

    with tf.variable_scope("loss"):
        if is_training:
            output_layer = tf.nn.dropout(output_layer, keep_prob=0.9)
        logits = tf.layers.dense(output_layer, num_labels, activation=None, 
                                 kernel_initializer=tf.truncated_normal_initializer(stddev=0.02))
        probabilities = tf.nn.softmax(logits)

        if labels is not None:
            one_hot_labels = tf.one_hot(labels, depth=num_labels, dtype=tf.float32)
            loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=labels, logits=logits)
            loss = tf.multiply(loss, tf.cast(input_mask, tf.float32))
            per_example_loss = tf.reduce_sum(loss, axis=-1)
            loss = tf.reduce_mean(per_example_loss)
        else:
            loss, per_example_loss = None, None
    return (loss, per_example_loss, logits, probabilities)

def model_fn_builder(bert_config, num_labels, init_checkpoint, learning_rate,
                     num_train_steps, num_warmup_steps, use_tpu,
                     use_one_hot_embeddings):

    def model_fn(features, labels, mode, params):
        tf.logging.info("*** Features ***")
        for name in sorted(features.keys()):
            tf.logging.info("  name = %s, shape = %s" % (name, features[name].shape))
        input_ids = features["input_ids"]
        input_mask = features["input_mask"]
        segment_ids = features["segment_ids"]
        label_ids = features["label_ids"]

        print('shape of input_ids', input_ids.shape)
        # label_mask = features["label_mask"]
        is_training = (mode == tf.estimator.ModeKeys.TRAIN)

        # 使用参数构建模型,input_idx 就是输入的样本idx表示，label_ids 就是标签的idx表示
        total_loss, logits, trans, pred_ids = create_model(
            bert_config, is_training, input_ids, input_mask, 
            segment_ids, label_ids, num_labels, False)

        tvars = tf.trainable_variables()
        # 加载BERT模型
        if init_checkpoint:
            (assignment_map, initialized_variable_names) = \
                 modeling.get_assignment_map_from_checkpoint(tvars,
                                                             init_checkpoint)
            tf.train.init_from_checkpoint(init_checkpoint, assignment_map)

        output_spec = None
        if mode == tf.estimator.ModeKeys.TRAIN:
            #train_op = optimizer.optimizer(total_loss, learning_rate, num_train_steps)
            train_op = optimization.create_optimizer(
                 total_loss, learning_rate, num_train_steps, num_warmup_steps, False)
            hook_dict = {}
            hook_dict['loss'] = total_loss
            hook_dict['global_steps'] = tf.train.get_or_create_global_step()
            logging_hook = tf.train.LoggingTensorHook(
                hook_dict, every_n_iter=100)

            output_spec = tf.estimator.EstimatorSpec(
                mode=mode,
                loss=total_loss,
                train_op=train_op,
                training_hooks=[logging_hook])

        elif mode == tf.estimator.ModeKeys.EVAL:
            # 针对NER ,进行了修改
            def metric_fn(label_ids, pred_ids):
                return {
                    "eval_loss": tf.metrics.mean_squared_error(labels=label_ids, predictions=pred_ids),
                }

            eval_metrics = metric_fn(label_ids, pred_ids)
            output_spec = tf.estimator.EstimatorSpec(
                mode=mode,
                loss=total_loss,
                eval_metric_ops=eval_metrics
            )
        else:
            output_spec = tf.estimator.EstimatorSpec(
                mode=mode,
                predictions=tf.argmax(pred_ids, axis=-1)
            )
        return output_spec

    return model_fn


def main():
    vocab_file = FLAGS.vocab_file
    tokenizer = tokenization.FullTokenizer(vocab_file=vocab_file, do_lower_case=FLAGS.do_lower_case)

    from dataset.msra_ner import MSRA_NER
    msra = MSRA_NER()
    label_list = msra.get_labels()
    train_examples = msra.get_train_examples()
    dev_examples = msra.get_dev_examples()
    test_examples = msra.get_test_examples()

    from utils.sentence_cutter import segment_sentence
    from collections import namedtuple
    def example_preprocess(examples):
        processed_examples = []
        NERItemClass = namedtuple('NERItemClass',['text', 'label'])
        for example in examples:
            for item in segment_sentence(example.text, example.label):
                processed_examples.append(NERItemClass(text=item[0], label=item[1]))
        return processed_examples

    train_examples = example_preprocess(train_examples)
    dev_examples = example_preprocess(dev_examples)
    test_examples = example_preprocess(test_examples)

    num_train_steps = int(len(train_examples) / FLAGS.train_batch_size * FLAGS.num_train_epochs)
    num_warmup_steps = int(num_train_steps * FLAGS.warmup_proportion)

    if not FLAGS.do_train and not FLAGS.do_eval:
        raise ValueError("At least one of `do_train` or `do_eval` must be True.")
    bert_config = modeling.BertConfig.from_json_file(FLAGS.bert_config_file)
    if FLAGS.max_seq_length > bert_config.max_position_embeddings:
        raise ValueError(
            "Cannot use sequence length %d because the BERT model "
            "was only trained up to sequence length %d" %
            (FLAGS.max_seq_length, bert_config.max_position_embeddings))

    if not tf.gfile.Exists(FLAGS.output_dir):
        tf.gfile.MakeDirs(FLAGS.output_dir)

    model_fn = model_fn_builder(
            bert_config=bert_config,
            num_labels=len(label_list),
            init_checkpoint=FLAGS.init_checkpoint,
            learning_rate=FLAGS.learning_rate,
            num_train_steps=num_train_steps,
            num_warmup_steps=num_warmup_steps,
            use_tpu=FLAGS.use_tpu,
            use_one_hot_embeddings=FLAGS.use_tpu)
    cfg = tf.estimator.RunConfig(save_checkpoints_steps=100, 
                                 save_summary_steps=100, log_step_count_steps=10)
    estimator = tf.estimator.Estimator(model_fn=model_fn,  
                                           model_dir=FLAGS.output_dir, 
                                           params={"batch_size": 32},
                                           config=cfg)
    if FLAGS.do_train:
        train_file = os.path.join(FLAGS.output_dir, "train.tf_record")    
        file_based_convert_examples_to_features(
            train_examples, label_list, FLAGS.max_seq_length, tokenizer, train_file)
        train_input_fn = file_based_input_fn_builder(
            input_file=train_file,
            seq_length=FLAGS.max_seq_length,
            is_training=True,
            drop_remainder=False)
        estimator.train(input_fn=train_input_fn, max_steps=num_train_steps)
    else:
        dev_file = os.path.join(FLAGS.output_dir, "dev.tf_record")    
        file_based_convert_examples_to_features(
            dev_examples, label_list, FLAGS.max_seq_length, tokenizer, dev_file)
        test_file = os.path.join(FLAGS.output_dir, "test.tf_record")    
        file_based_convert_examples_to_features(
            test_examples, label_list, FLAGS.max_seq_length, tokenizer, test_file)
        
        dev_input_fn = file_based_input_fn_builder(
            input_file=dev_file,
            seq_length=FLAGS.max_seq_length,
            is_training=False,
            drop_remainder=False)
        test_input_fn = file_based_input_fn_builder(
            input_file=test_file,
            seq_length=FLAGS.max_seq_length,
            is_training=False,
            drop_remainder=False)
        dev_preds_output = estimator.predict(input_fn=dev_input_fn)
        test_preds_output = estimator.predict(input_fn=test_input_fn)
        
        def evaluate_ner(examples, preds_output):
            from utils.evaluate import evaluate_with_conlleval
            texts = [example.text for example in examples]
            golds = [example.label for example in examples]
            preds = []
            for idx, pred in enumerate(preds_output):
                gold_length = len(golds[idx])
                pred = [label_list[int(x)] for x in pred[1:gold_length+1]]
                if gold_length != len(pred):
                    pred = pred + ["O" for _ in range(gold_length - len(pred))]
                preds.append(pred)
            eval_lines = evaluate_with_conlleval(texts, golds, preds, 
                        os.path.join(FLAGS.output_dir, "ner_predict.txt"))

            for line in eval_lines:
                tf.logging.info(line)

        evaluate_ner(dev_examples, dev_preds_output)
        evaluate_ner(test_examples, test_preds_output)


if __name__ == "__main__":
    main()
