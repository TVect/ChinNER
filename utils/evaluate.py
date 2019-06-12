import os
from utils.conlleval import return_report

def test_ner(results, output_file):
    """
    Run perl script to evaluate model
    """
    with open(output_file, "w") as f:
        to_write = []
        for block in results:
            for line in block:
                to_write.append(line + "\n")
            to_write.append("\n")

        f.writelines(to_write)
    eval_lines = return_report(output_file)
    return eval_lines


def evaluate_with_conlleval(texts, golds, preds, output_file):
    assert len(texts) == len(golds) == len(preds), \
           "len(texts):{}, len(golds):{}, len(preds):{}".format(len(texts), len(golds), len(preds))
    results = []
    for text, gold, pred in zip(texts, golds, preds):
        assert len(text) == len(gold) == len(pred), \
               "len(text):{}, len(gold):{}, len(pred):{}".format(len(text), len(gold), len(pred))
        result = [" ".join([char_item, gold_item, pred_item]) 
                  for char_item, gold_item, pred_item in zip(text, gold, pred)]
        results.append(result)
    return test_ner(results, output_file)

