#!/bin/bash

cd ..

python -m models.bert_ner.main \
	--do_lower_case=False \
    --use_crf=False    \
    --max_seq_length=128   \
    --batch_size=32   \
    --learning_rate=2e-5   \
    --num_train_epochs=4   \
    --output_dir=./models/bert_ner/output
