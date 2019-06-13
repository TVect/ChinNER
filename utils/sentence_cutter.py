
def segment_sentence(text, label, max_length=125):
    punc_tokens = ["。", "？", "！", "?", "，", "℃", "；", "、", "："]
    if len(text) <= max_length:
        return [[text, label]]
    punc_id = None
    for i in range(max_length):
        wd = text[max_length - i - 1]
        if wd in punc_tokens:
            punc_id = max_length - i - 1
            break
    if punc_id != None: 
        ret = [[text[: punc_id+1], label[: punc_id+1]]]
        ret.extend(segment_sentence(text[punc_id+1:], label[punc_id+1:]))
        return ret
    else:
        ret = [[text[: max_length], label[: max_length]]]
        ret.extend(segment_sentence(text[max_length:], label[max_length:]))
        return ret


if __name__ == '__main__':
    from dataset.msra_ner import MSRA_NER
    ds = MSRA_NER(tagging_schema="iobes")
    import collections
    cnt = collections.Counter()
    for example in ds.get_test_examples():
        cnt.update(len(item[0]) for item in segment_sentence(example.text, example.label))
        # print("".join(example.text))
        for item in segment_sentence(example.text, example.label):
            if len(item[0]) < 10:
                print("".join(item[0]))
            # print("".join(item[0]))
            # print("".join(item[1]))

