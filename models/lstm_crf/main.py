import os
import numpy as np
import tensorflow as tf
import codecs
from models.lstm_crf import data_helper
from models.lstm_crf.model import model_fn
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
              "clip": FLAGS.clip,
              "steps_logging": FLAGS.steps_logging}
    
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
    
    data_helper.file_based_convert_examples_to_features(
            train_examples, emb_tokens, label_list, lower=FLAGS.lower,
            output_file=os.path.join(FLAGS.ckpt_path, "train.tf_record"))
    data_helper.file_based_convert_examples_to_features(
            dev_examples, emb_tokens, label_list, lower=FLAGS.lower,
            output_file=os.path.join(FLAGS.ckpt_path,"dev.tf_record"))
    data_helper.file_based_convert_examples_to_features(
            test_examples, emb_tokens, label_list, lower=FLAGS.lower,
            output_file=os.path.join(FLAGS.ckpt_path,"test.tf_record"))
    
    train_input_fn = data_helper.file_based_input_fn_builder(
            input_file=os.path.join(FLAGS.ckpt_path, "train.tf_record"), 
            is_training=True, batch_size=FLAGS.batch_size)
    dev_input_fn = data_helper.file_based_input_fn_builder(
            input_file=os.path.join(FLAGS.ckpt_path, "dev.tf_record"), 
            is_training=False, batch_size=FLAGS.batch_size)
    test_input_fn = data_helper.file_based_input_fn_builder(
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
