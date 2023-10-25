import sys
sys.path.append("/data3/chenziang/codes/Bert-VITS2")
if __name__ == "__main__":
    from text.chinese_bert import get_bert_feature
    from text.chinese import g2p, text_normalize
    text = "由于人们之间普遍以自然语言的形式交流，互联网和各种数据库中的这些非结构化数据中蕴藏了价值连城的大量信息。"
    text = text_normalize(text)
    print(text)
    phones, tones, word2ph = g2p(text)
    bert = get_bert_feature(text, word2ph)

    # print(phones)
    # print(len(phones))
    # print(tones)
    # print(len(tones))
    # print(word2ph)
    # print(len(word2ph))
    print(bert.shape)