import pickle
import numpy as np
import tensorflow as tf
import tf_metrics
import functools
import data_api
from tensorflow.python.training.training_util import global_step

tf.logging.set_verbosity(tf.logging.INFO)

flags = tf.app.flags
flags.DEFINE_integer("seg_dim", 20, "Embedding size for segmentation, 0 if not used")
flags.DEFINE_integer("char_dim", 100, "Embedding size for characters")
flags.DEFINE_string("tag_schema", "iobes", "tagging schema iobes or iob")
flags.DEFINE_integer("hidden_units", 32, "hidden units")
flags.DEFINE_integer("num_lstm_layers", 1, "num of lstm layers")

# configurations for training
flags.DEFINE_float("clip", 5.0, "Gradient clip")
flags.DEFINE_float("dropout", 0.5, "Dropout rate")
flags.DEFINE_float("learning_rate", 0.001, "Initial learning rate")
flags.DEFINE_string("optimizer", "adam", "Optimizer for training")
flags.DEFINE_boolean("pre_emb", True, "Wither use pre-trained embedding")
flags.DEFINE_boolean("zeros", False, "Wither replace digits with zero")
flags.DEFINE_boolean("lower", True, "Wither lower case")

flags.DEFINE_integer("max_epoch", 200, "maximum training epochs")
flags.DEFINE_integer("steps_check", 100, "steps per checkpoint")
flags.DEFINE_integer("steps_summary", 20, "steps per summary")
flags.DEFINE_integer("steps_logging", 20, "steps per summary")
flags.DEFINE_integer("batch_size", 32, "batch size")
# flags.DEFINE_integer("epochs_between_evals", 1, "The number of training epochs to run between evaluations.")

flags.DEFINE_string("ckpt_path", "./checkpoints_bilstm", "Path to save model")
flags.DEFINE_string("log_file", "train.log", "File for log")
flags.DEFINE_string("map_file", "maps.pkl", "file for maps")
flags.DEFINE_string("script", "conlleval", "evaluation script")
flags.DEFINE_string("result_path", "result", "Path for results")
flags.DEFINE_string("emb_file", "wiki_100.utf8", "Path for pre_trained embedding")
flags.DEFINE_string("data_dir", "../dataset/msra", "Path for train data")

FLAGS = tf.app.flags.FLAGS

train_data, dev_data, test_data = data_api.load_msra_data(data_dir=FLAGS.data_dir, 
                                                          flag_lower=FLAGS.lower,
                                                          flag_zeros=FLAGS.zeros,
                                                          flag_tag_schema=FLAGS.tag_schema,
                                                          flag_map_file=FLAGS.map_file, 
                                                          flag_emb_file=FLAGS.emb_file)

with open(FLAGS.map_file, "rb") as f:
    char_to_id, id_to_char, tag_to_id, id_to_tag = pickle.load(f)

def input_fn(in_data):
    def data_generator(datas):
        for item in datas:
            yield item[1], item[2], item[3]

    data = tf.data.Dataset.from_generator(generator=functools.partial(data_generator, in_data), 
                                          output_types=(tf.int32, tf.int32, tf.int32))
    data = data.shuffle(10000).repeat(1).padded_batch(batch_size=FLAGS.batch_size, 
                                                      padded_shapes=([None], [None], [None]))

    iterator = data.make_one_shot_iterator()
    chars, segs, tags = iterator.get_next()
    features = {"chars": chars, "segs": segs}
    labels = {"tags": tags}  # Could be None in your predict_input_fn
    labels = tags
    return features, labels

def model_fn(features, labels, mode, params):
    '''
    @param features: This is the x-arg from the input_fn
    @param labels: This is the y-arg from the input_fn
    @param mode: TRAIN | EVAL | PREDICT
    @param params: User-defined hyper-parameters, e.g. learning-rate
    '''
    char_inputs = features["chars"]
    seg_inputs = features["segs"]
#     in_lengths = features["in_lengths"]
    length = tf.reduce_sum(tf.sign(tf.abs(char_inputs)), reduction_indices=1)
    in_lengths = tf.cast(length, tf.int32)

    char_lookup = tf.get_variable(name="char_embedding", 
                                  shape=[params["num_chars"], params["char_dim"]], 
                                  initializer=tf.contrib.layers.xavier_initializer())
    char_emb = tf.nn.embedding_lookup(char_lookup, char_inputs)
    seg_lookup = tf.get_variable(name="seg_embedding", 
                                 shape=[params["num_segs"], params["seg_dim"]], 
                                 initializer=tf.contrib.layers.xavier_initializer())
    seg_emb = tf.nn.embedding_lookup(seg_lookup, seg_inputs)
    embed = tf.concat([char_emb, seg_emb], axis=-1)

    lstm_input = tf.layers.dropout(embed, rate=params["dropout_rate"])
    def gen_lstm_cell():
        return tf.nn.rnn_cell.DropoutWrapper(
            tf.contrib.rnn.LSTMCell(num_units=int((params["char_dim"] + params["seg_dim"]) / 2), 
                                    use_peepholes=True), 
            output_keep_prob=params["rnn_dropout_rate"])
    lstm_output, _, _ = tf.contrib.rnn.stack_bidirectional_dynamic_rnn(
        [gen_lstm_cell() for _ in range(params["num_lstm_layers"])], 
        [gen_lstm_cell() for _ in range(params["num_lstm_layers"])], 
        inputs=lstm_input, 
        sequence_length=in_lengths,
        dtype=tf.float32)
    dense_hidden = tf.layers.dense(lstm_output, params["hidden_units"], activation=tf.nn.relu)
    dense_output = tf.layers.dense(dense_hidden, params["num_tags"])
    
    transition_params = tf.get_variable(name="transitions", 
                                        shape=[params["num_tags"], params["num_tags"]])
    predictions, _ = tf.contrib.crf.crf_decode(dense_output, transition_params, in_lengths)
    
    if mode == tf.estimator.ModeKeys.PREDICT:
        spec = tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)
    else:
        log_likelihood, _ = tf.contrib.crf.crf_log_likelihood(
            dense_output, labels, in_lengths, transition_params=transition_params)
        loss = tf.reduce_mean(-log_likelihood)
        weights = tf.sequence_mask(in_lengths)
        num_tags = params["num_tags"]
        indices = [id for tag, id in tag_to_id.items() if tag != 'O']
        metrics = {
            'accuracy': tf.metrics.accuracy(labels, predictions, weights), 
            'precision': tf_metrics.precision(labels, predictions, num_tags, indices, weights), 
            'recall': tf_metrics.recall(labels, predictions, num_tags, indices, weights), 
            'f1': tf_metrics.f1(labels, predictions, num_tags, indices, weights)
            }
        if mode == tf.estimator.ModeKeys.EVAL:
            spec = tf.estimator.EstimatorSpec(mode=mode, loss=loss, eval_metric_ops=metrics)
        elif mode == tf.estimator.ModeKeys.TRAIN:
            train_op = tf.train.AdamOptimizer(learning_rate=params["learning_rate"])\
                    .minimize(loss, global_step=tf.train.get_or_create_global_step())
            logging_hook = tf.train.LoggingTensorHook(
                {"loss" : loss, 
                 "accuracy" : metrics["accuracy"][1], 
                 "precision": metrics["precision"][1],
                 "recall": metrics["recall"][1],
                 "f1": metrics["f1"][1]}, 
                every_n_iter=FLAGS.steps_logging)
            
            tf.summary.scalar('accuracy', metrics["accuracy"][1])
            tf.summary.scalar('precision', metrics["precision"][1])
            tf.summary.scalar('recall', metrics["recall"][1])
            tf.summary.scalar('f1', metrics["f1"][1])
            
            spec = tf.estimator.EstimatorSpec(mode=mode, loss=loss, 
                                              train_op=train_op, 
                                              eval_metric_ops=metrics, 
                                              training_hooks = [logging_hook])

    return spec

params = {"num_chars": len(char_to_id), 
          "num_segs": 4, 
          "char_dim": len(tag_to_id), 
          "seg_dim": FLAGS.seg_dim, 
          "dropout_rate": FLAGS.dropout, 
          "rnn_dropout_rate": FLAGS.dropout, 
          "num_lstm_layers": FLAGS.num_lstm_layers, 
          "hidden_units": FLAGS.hidden_units, 
          "num_tags": len(tag_to_id), 
          "learning_rate": FLAGS.learning_rate}

cfg = tf.estimator.RunConfig(save_checkpoints_steps=FLAGS.steps_check, 
                             save_summary_steps=FLAGS.steps_summary)
model = tf.estimator.Estimator(model_fn=model_fn, 
                               params=params, 
                               model_dir=FLAGS.ckpt_path, 
                               config=cfg)

# Train and evaluate model.
for epoch_id in range(FLAGS.max_epoch):
    tf.logging.info("================= epoch_id:{} =================".format(epoch_id))
    model.train(input_fn=functools.partial(input_fn, in_data=train_data))
    eval_results = model.evaluate(input_fn=functools.partial(input_fn, in_data=dev_data))
    print('\nEvaluation results:\n\t%s\n' % eval_results)
