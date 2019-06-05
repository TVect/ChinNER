import os
import pickle
import itertools
from sklearn.model_selection import train_test_split
from loader import load_sentences, update_tag_scheme
from loader import char_mapping, tag_mapping
from loader import augment_with_pretrained, prepare_dataset
# from utils import get_logger, make_path, clean, create_model, save_model
# from utils import print_config, save_config, load_config, test_ner
# from data_utils import load_word2vec, create_input, input_from_line, BatchManager

import tensorflow as tf

def load_msra_data(data_dir, 
                   flag_lower=True, 
                   flag_zeros=False, 
                   flag_tag_schema="iobes", 
                   flag_map_file="maps.pkl", 
                   flag_emb_file="wiki_100.utf8"):
    # load data sets
    train_file = os.path.join(data_dir, "msra_train_bio")
    test_file = os.path.join(data_dir, "msra_test_bio")
    train_sentences = load_sentences(train_file, flag_lower, flag_zeros)
    train_sentences, dev_sentences = train_test_split(train_sentences, test_size=3000, random_state=17)
    test_sentences = load_sentences(test_file, flag_lower, flag_zeros)

    # Use selected tagging scheme (IOB / IOBES)
    update_tag_scheme(train_sentences, flag_tag_schema)
    update_tag_scheme(dev_sentences, flag_tag_schema)
    update_tag_scheme(test_sentences, flag_tag_schema)

    # create maps if not exist
    if not os.path.isfile(flag_map_file):
        # create dictionary for word
        dico_chars_train = char_mapping(train_sentences, flag_lower)[0]
        dico_chars, char_to_id, id_to_char = augment_with_pretrained(
            dico_chars_train.copy(),
            flag_emb_file,
            list(itertools.chain.from_iterable(
                [[w[0] for w in s] for s in test_sentences])
            )
        )

        # Create a dictionary and a mapping for tags
        _t, tag_to_id, id_to_tag = tag_mapping(train_sentences)
        with open(flag_map_file, "wb") as f:
            pickle.dump([char_to_id, id_to_char, tag_to_id, id_to_tag], f)
    else:
        with open(flag_map_file, "rb") as f:
            char_to_id, id_to_char, tag_to_id, id_to_tag = pickle.load(f)

    # prepare data, get a collection of list containing index
    train_data = prepare_dataset(train_sentences, char_to_id, tag_to_id, flag_lower)
    dev_data = prepare_dataset(dev_sentences, char_to_id, tag_to_id, flag_lower)
    test_data = prepare_dataset(test_sentences, char_to_id, tag_to_id, flag_lower)
    return train_data, dev_data, test_data
