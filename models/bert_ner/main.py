import os
import tensorflow as tf
from .bert import tokenization, modeling
from .model import model_fn_builder
from .data_helper import file_based_convert_examples_to_features
from .data_helper import file_based_input_fn_builder


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

flags.DEFINE_bool("use_crf", False, "Whether to use crf_layer")
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
            use_crf=FLAGS.use_crf)
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
