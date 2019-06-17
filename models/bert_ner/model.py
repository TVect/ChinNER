# -*- coding: utf-8 -*-

import tf_metrics
import tensorflow as tf
from .bert import modeling, optimization

def create_model(bert_config, is_training, input_ids, input_mask,
                 segment_ids, labels, num_labels, use_crf):
    model = modeling.BertModel(
        config=bert_config,
        is_training=is_training,
        input_ids=input_ids,
        input_mask=input_mask,
        token_type_ids=segment_ids)

    output_layer = model.get_sequence_output()

    with tf.variable_scope("loss"):
        if is_training:
            output_layer = tf.nn.dropout(output_layer, keep_prob=0.9)
        logits = tf.layers.dense(output_layer, num_labels, activation=None, 
                                 kernel_initializer=tf.truncated_normal_initializer(stddev=0.02))
        if use_crf:
            transition_params = tf.get_variable(name="transitions", 
                                        shape=[num_labels, num_labels],       
                                        initializer=tf.contrib.layers.xavier_initializer())
            in_lengths = tf.reduce_sum(input_mask, axis=-1)
            predictions, _ = tf.contrib.crf.crf_decode(logits, transition_params, in_lengths)
            
            if labels is not None:
                log_likelihood, _ = tf.contrib.crf.crf_log_likelihood(
                    logits, labels, in_lengths, transition_params=transition_params)
                loss = tf.reduce_mean(-log_likelihood)
            else:
                loss = None
        else:
            # probabilities = tf.nn.softmax(logits)
            predictions = tf.argmax(logits, axis=-1)
            if labels is not None:
                loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=labels, logits=logits)
                loss = tf.multiply(loss, tf.cast(input_mask, tf.float32))
                loss = tf.reduce_mean(tf.reduce_sum(loss, axis=-1))
            else:
                loss = None
    return (loss, predictions)

def model_fn_builder(bert_config, num_labels, init_checkpoint, learning_rate,
                     num_train_steps, num_warmup_steps, use_crf):

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
        total_loss, pred_ids = create_model(
            bert_config, is_training, input_ids, input_mask, 
            segment_ids, label_ids, num_labels, use_crf=use_crf)

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
            indices = list(range(1, num_labels))
            metrics = {
                'accuracy': tf.metrics.accuracy(labels, pred_ids, input_mask), 
                'precision': tf_metrics.precision(labels, pred_ids, num_labels, indices, input_mask), 
                'recall': tf_metrics.recall(labels, pred_ids, num_labels, indices, input_mask), 
                'f1': tf_metrics.f1(labels, pred_ids, num_labels, indices, input_mask)
            }

            output_spec = tf.estimator.EstimatorSpec(
                mode=mode,
                loss=total_loss,
                eval_metric_ops=metrics
            )
        else:
            output_spec = tf.estimator.EstimatorSpec(
                mode=mode,
                predictions=pred_ids
            )
        return output_spec

    return model_fn
