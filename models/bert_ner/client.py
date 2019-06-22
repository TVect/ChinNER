# -*- coding: utf-8 -*-

import os
import collections
import tensorflow as tf
from .data_helper import DataProcessor

class ServingClient:
    
    NERItemClass = collections.namedtuple('NERItemClass',['text', 'label'])
    
    def __init__(self, model_path, data_processor):
        self.predict_fn = tf.contrib.predictor.from_saved_model(model_path)
        self.data_processor = data_processor

    def predict(self, in_texts):
        '''
        @param in_texts: list of tokens, [[tok1, tok2, ...], [tok2, tok3, ...]]
        '''
        max_length = max(len(text) for text in in_texts)
        assert max_length + 2 <= self.data_processor.max_seq_length, "text is too long"

        examples = [self.NERItemClass(text=text, label=None) for text in in_texts]
        input_features = [self.data_processor.convert_example_to_features(example).SerializeToString() 
                            for example in examples]
        rets = self.predict_fn({"examples": input_features})
        ret_tags = [self.data_processor.convert_ids_to_labels(ret)[1:len(in_texts[idx])+1] 
                    for idx, ret in enumerate(rets["output"])]
        return ret_tags

if __name__ == "__main__":
    FILE_HOME = os.path.abspath(os.path.dirname(__file__))
    model_path = os.path.join(FILE_HOME, "./output/1560850786")
    processor_file = os.path.join(FILE_HOME, "./output/data_processor.json")
    data_processor = DataProcessor.load_from_file(processor_file)
    
    texts = [list("老一代红军将领孙毅将军为我们展示的北京市地图")]
    client = ServingClient(model_path, data_processor)
    ret_tags = client.predict(texts)
    for text, tag in zip(texts, ret_tags): 
        for item in zip(texts, ret_tags): 
            print([pair for pair in zip(*item)])
