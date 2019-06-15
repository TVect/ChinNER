# -*- coding: utf-8 -*-
import tf_metrics
import tensorflow as tf

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
    # lstm_output, output_state_fw, output_state_bw = tf.contrib.rnn.stack_bidirectional_dynamic_rnn(
    #     [gen_lstm_cell() for _ in range(params["num_lstm_layers"])], 
    #     [gen_lstm_cell() for _ in range(params["num_lstm_layers"])], 
    #     inputs=lstm_input, 
    #     sequence_length=in_lengths,
    #     dtype=tf.float32)
    fw_cell = tf.contrib.rnn.CoupledInputForgetGateLSTMCell(
            100, use_peepholes=True, initializer=tf.contrib.layers.xavier_initializer(), state_is_tuple=True)
    bw_cell = tf.contrib.rnn.CoupledInputForgetGateLSTMCell(
            100, use_peepholes=True, initializer=tf.contrib.layers.xavier_initializer(), state_is_tuple=True)
    outputs, final_states = tf.nn.bidirectional_dynamic_rnn(
            fw_cell, bw_cell, lstm_input, dtype=tf.float32, sequence_length=in_lengths)
    lstm_output = tf.concat(outputs, axis=2)

    dense_hidden = tf.layers.dense(
            lstm_output, 100, activation=tf.nn.tanh, kernel_initializer=tf.contrib.layers.xavier_initializer())
    dense_output = tf.layers.dense(
            dense_hidden, params["num_tags"], kernel_initializer=tf.contrib.layers.xavier_initializer())
    
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
                every_n_iter=params["steps_logging"])
            
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
