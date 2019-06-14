#!/bin/bash

cd ..

python -m models.bert_ner.main \
	--do_lower_case=False \
    --do_train=True   \
    --do_eval=True   \
    --do_predict=True \
    --data_dir=data   \
    --use_crf=False    \
    --max_seq_length=128   \
    --train_batch_size=32   \
    --learning_rate=2e-5   \
    --num_train_epochs=4.0   \
    --output_dir=./models/bert_ner/output