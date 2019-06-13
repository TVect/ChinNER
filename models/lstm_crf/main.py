import os
import numpy as np
import tensorflow as tf
import tf_metrics
import functools
import codecs
from models.lstm_crf import data_api
from dataset.msra_ner import MSRA_NER
from utils.evaluate import evaluate_with_conlleval

FILE_HOME = os.path.abspath(os.path.dirname(__file__))

tf.logging.set_verbosity(tf.logging.INFO)
tf.logging.propagate = False

flags = tf.app.flags
flags.DEFINE_integer("seg_dim", 20, "Embedding size for segmentation, 0 if not used")
flags.DEFINE_integer("char_dim", 100, "Embedding size for characters")
flags.DEFINE_string("tag_schema", "iobes", "tagging schema iobes or iob")
flags.DEFINE_integer("hidden_units", 32, "hidden units")
flags.DEFINE_integer("num_lstm_layers", 2, "num of lstm layers")

# configurations for training
flags.DEFINE_float("clip", 5.0, "Gradient clip")
flags.DEFINE_float("dropout", 0.5, "Dropout rate")
flags.DEFINE_float("learning_rate", 0.001, "Initial learning rate")
flags.DEFINE_string("optimizer", "adam", "Optimizer for training")
flags.DEFINE_boolean("pre_emb", True, "Wither use pre-trained embedding")
flags.DEFINE_boolean("zeros", False, "Wither replace digits with zero")
flags.DEFINE_boolean("lower", True, "Wither lower case")

flags.DEFINE_integer("max_epoch", 200, "maximum training epochs")
flags.DEFINE_integer("steps_check", 1000, "steps per checkpoint")
flags.DEFINE_integer("steps_summary", 100, "steps per summary")
flags.DEFINE_integer("steps_logging", 100, "steps per summary")
flags.DEFINE_integer("batch_size", 32, "batch size")
# flags.DEFINE_integer("epochs_between_evals", 1, "The number of training epochs to run between evaluations.")

flags.DEFINE_string("ckpt_path", 
                    os.path.join(FILE_HOME, "ckpt_bilstm"), 
                    "Path to save model")
flags.DEFINE_string("log_file", 
                    os.path.join(FILE_HOME, "train.log"), 
                    "File for log")
flags.DEFINE_string("map_file", 
                    os.path.join(FILE_HOME, "maps.pkl"), 
                    "file for maps")
flags.DEFINE_string("emb_file", 
                    os.path.join(FILE_HOME, "wiki_100.utf8"), 
                    "Path for pre_trained embedding")


FLAGS = tf.app.flags.FLAGS

def model_fn(features, labels, mode, params):
    '''
    @param features: This is the x-arg from the input_fn
    @param labels: This is the y-arg from the input_fn
    @param mode: TRAIN | EVAL | PREDICT
    @param params: User-defined hyper-parameters, e.g. learning-rate
    '''
    char_inputs = features["char_ids"]
    seg_inputs = features["seg_ids"]
#     in_lengths = features["in_lengths"]
    length = tf.reduce_sum(tf.sign(tf.abs(char_inputs)), reduction_indices=1)
    in_lengths = tf.cast(length, tf.int32)
    
    if params["pre_emb"]:
        char_lookup = tf.Variable(params["emb_matrix"], name="char_embedding")
    else:
        char_lookup = tf.get_variable(name="char_embedding", 
                                  shape=[params["num_chars"], params["char_dim"]], 
                                  initializer=tf.contrib.layers.xavier_initializer())
    char_emb = tf.nn.embedding_lookup(char_lookup, char_inputs)
    seg_lookup = tf.get_variable(name="seg_embedding", 
                                 shape=[params["num_segs"], params["seg_dim"]], 
                                 initializer=tf.contrib.layers.xavier_initializer())
    seg_emb = tf.nn.embedding_lookup(seg_lookup, seg_inputs)
    embed = tf.concat([char_emb, seg_emb], axis=-1)
    
    is_training = (mode == tf.estimator.ModeKeys.TRAIN)
    lstm_input = tf.layers.dropout(embed, rate=params["dropout_rate"], 
                                   training=is_training)
    
    def gen_lstm_cell():
        return tf.nn.rnn_cell.DropoutWrapper(
            tf.contrib.rnn.LSTMCell(num_units=int((params["char_dim"] + params["seg_dim"]) / 2), 
                                    initializer=tf.contrib.layers.xavier_initializer(),
                                    use_peepholes=True), 
            output_keep_prob=params["rnn_dropout_rate"])
    lstm_output, output_state_fw, output_state_bw = tf.contrib.rnn.stack_bidirectional_dynamic_rnn(
        [gen_lstm_cell() for _ in range(params["num_lstm_layers"])], 
        [gen_lstm_cell() for _ in range(params["num_lstm_layers"])], 
        inputs=lstm_input, 
        sequence_length=in_lengths,
        dtype=tf.float32)
    # fw_cell = tf.contrib.rnn.CoupledInputForgetGateLSTMCell(100, use_peepholes=True, initializer=tf.contrib.layers.xavier_initializer(), state_is_tuple=True)
    # bw_cell = tf.contrib.rnn.CoupledInputForgetGateLSTMCell(100, use_peepholes=True, initializer=tf.contrib.layers.xavier_initializer(), state_is_tuple=True)
    # outputs, final_states = tf.nn.bidirectional_dynamic_rnn(fw_cell, bw_cell, lstm_input, dtype=tf.float32, sequence_length=in_lengths)
    # lstm_output = tf.concat(outputs, axis=2)

    dense_hidden = tf.layers.dense(lstm_output, 100, activation=tf.nn.tanh, kernel_initializer=tf.contrib.layers.xavier_initializer())
    dense_output = tf.layers.dense(dense_hidden, params["num_tags"], kernel_initializer=tf.contrib.layers.xavier_initializer())
    
    transition_params = tf.get_variable(name="transitions", 
                                        shape=[params["num_tags"], params["num_tags"]], 
                                        initializer=tf.contrib.layers.xavier_initializer())
    predictions, _ = tf.contrib.crf.crf_decode(dense_output, transition_params, in_lengths)
    
    if mode == tf.estimator.ModeKeys.PREDICT:
        spec = tf.estimator.EstimatorSpec(mode=mode, 
                                          predictions={"preds": predictions})
    else:
        log_likelihood, _ = tf.contrib.crf.crf_log_likelihood(
            dense_output, labels, in_lengths, transition_params=transition_params)
        loss = tf.reduce_mean(-log_likelihood)
        weights = tf.sequence_mask(in_lengths)
        num_tags = params["num_tags"]
        indices = list(range(1, num_tags))
        metrics = {
            'accuracy': tf.metrics.accuracy(labels, predictions, weights), 
            'precision': tf_metrics.precision(labels, predictions, num_tags, indices, weights), 
            'recall': tf_metrics.recall(labels, predictions, num_tags, indices, weights), 
            'f1': tf_metrics.f1(labels, predictions, num_tags, indices, weights)
            }
        if mode == tf.estimator.ModeKeys.EVAL:
            spec = tf.estimator.EstimatorSpec(mode=mode, loss=loss, eval_metric_ops=metrics)
        elif mode == tf.estimator.ModeKeys.TRAIN:
            train_op = tf.train.AdamOptimizer(learning_rate=params["learning_rate"])
            train_op = tf.contrib.estimator.clip_gradients_by_norm(train_op, params["clip"])
            train_op = train_op.minimize(loss, global_step=tf.train.get_or_create_global_step())
            logging_hook = tf.train.LoggingTensorHook(
                {"loss" : loss, 
                 "accuracy" : metrics["accuracy"][1], 
                 "precision": metrics["precision"][1],
                 "recall": metrics["recall"][1],
                 "f1": metrics["f1"][1]}, 
                every_n_iter=FLAGS.steps_logging)
            
            '''
            tf.summary.scalar('accuracy', metrics["accuracy"][1])
            tf.summary.scalar('precision', metrics["precision"][1])
            tf.summary.scalar('recall', metrics["recall"][1])
            tf.summary.scalar('f1', metrics["f1"][1])
            '''
            spec = tf.estimator.EstimatorSpec(mode=mode, loss=loss, 
                                              train_op=train_op, 
                                              eval_metric_ops=metrics, 
                                              training_hooks = [logging_hook])

    return spec


def main(args):
    if not tf.gfile.Exists(FLAGS.ckpt_path):
        tf.gfile.MakeDirs(FLAGS.ckpt_path)

    msra = MSRA_NER(tagging_schema="iobes")
    train_examples = msra.get_train_examples()
    dev_examples = msra.get_dev_examples()
    test_examples = msra.get_test_examples()
    
    with codecs.open(FLAGS.emb_file, mode='r', encoding="utf8") as fr:
        _, dim = fr.readline().split()
        emb_tokens = ["<PAD>", "<UNK>"]    # 计算句子长度的时候有用.
        emb_matrix = [np.random.randn(int(dim)), np.random.randn(int(dim))]
        contents = [line.strip().split() for line in fr]
        emb_tokens.extend([content[0] for content in contents])
        emb_matrix.extend([list(map(float, content[1:])) for content in contents])
        emb_matrix = np.array(emb_matrix)

    params = {"num_chars": len(emb_tokens), 
              "num_segs": 4,  
              "char_dim": FLAGS.char_dim,
              "seg_dim": FLAGS.seg_dim, 
              "dropout_rate": FLAGS.dropout, 
              "rnn_dropout_rate": FLAGS.dropout, 
              "num_lstm_layers": FLAGS.num_lstm_layers, 
              "hidden_units": FLAGS.hidden_units, 
              "num_tags": msra.num_labels, 
              "learning_rate": FLAGS.learning_rate,
              "clip": FLAGS.clip}
    
    if FLAGS.pre_emb:
        params["emb_matrix"] = emb_matrix.astype(np.float32)
        params["pre_emb"] = True
    
    cfg = tf.estimator.RunConfig(save_checkpoints_steps=FLAGS.steps_check, 
                                 save_summary_steps=FLAGS.steps_summary, 
                                 log_step_count_steps=FLAGS.steps_logging)
    model = tf.estimator.Estimator(model_fn=model_fn, 
                                   params=params, 
                                   model_dir=FLAGS.ckpt_path, 
                                   config=cfg)
    label_list = msra.get_labels()
    
    data_api.file_based_convert_examples_to_features(
            train_examples, emb_tokens, label_list, lower=FLAGS.lower,
            output_file=os.path.join(FLAGS.ckpt_path, "train.tf_record"))
    data_api.file_based_convert_examples_to_features(
            dev_examples, emb_tokens, label_list, lower=FLAGS.lower,
            output_file=os.path.join(FLAGS.ckpt_path,"dev.tf_record"))
    data_api.file_based_convert_examples_to_features(
            test_examples, emb_tokens, label_list, lower=FLAGS.lower,
            output_file=os.path.join(FLAGS.ckpt_path,"test.tf_record"))
    
    train_input_fn = data_api.file_based_input_fn_builder(
            input_file=os.path.join(FLAGS.ckpt_path, "train.tf_record"), 
            is_training=True, batch_size=FLAGS.batch_size)
    dev_input_fn = data_api.file_based_input_fn_builder(
            input_file=os.path.join(FLAGS.ckpt_path, "dev.tf_record"), 
            is_training=False, batch_size=FLAGS.batch_size)
    test_input_fn = data_api.file_based_input_fn_builder(
            input_file=os.path.join(FLAGS.ckpt_path, "test.tf_record"), 
            is_training=False, batch_size=FLAGS.batch_size)
   
    def evaluate_ner(examples, preds_output):
        texts = [example.text for example in examples]
        golds = [example.label for example in examples]
        preds = []
        for idx, pred in enumerate(preds_output):
            gold_length = len(golds[idx])
            pred = [label_list[int(x)] for x in pred["preds"][:gold_length]]
            preds.append(pred)
        eval_lines = evaluate_with_conlleval(texts, golds, preds, 
                        os.path.join(FLAGS.ckpt_path, "ner_predict.txt"))

        for line in eval_lines:
            tf.logging.info(line)

    # Train and evaluate model.
    for epoch_id in range(FLAGS.max_epoch):
        tf.logging.info("================= epoch_id:{} =================".format(epoch_id))
        model.train(input_fn=train_input_fn)
        eval_results = model.evaluate(input_fn=dev_input_fn)
        tf.logging.info('\nEvaluation results:\n\t%s\n' % eval_results)

        evaluate_ner(dev_examples, model.predict(input_fn=dev_input_fn))
        evaluate_ner(test_examples, model.predict(input_fn=test_input_fn))


if __name__ == "__main__":
    tf.app.run()
