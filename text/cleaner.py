import sys
sys.path.append("/data3/chenziang/codes/Bert-VITS2")
from text import chinese, japanese, english, cleaned_text_to_sequence

language_module_map = {"ZH": chinese, "JP": japanese, "EN": english}
# language_module_map = {"ZH": chinese}


def clean_text(text, language):
    language_module = language_module_map[language]
    # 这里规范了数字、嗯呣等文本
    norm_text = language_module.text_normalize(text)
    # 音素、音调、每个字对应的音素数量
    phones, tones, word2ph = language_module.g2p(norm_text)
    return norm_text, phones, tones, word2ph

# 好像没被用到


def clean_text_bert(text, language):
    language_module = language_module_map[language]
    norm_text = language_module.text_normalize(text)
    phones, tones, word2ph = language_module.g2p(norm_text)
    bert = language_module.get_bert_feature(norm_text, word2ph)
    return phones, tones, bert


def text_to_sequence(text, language):
    norm_text, phones, tones, word2ph = clean_text(text, language)
    return cleaned_text_to_sequence(phones, tones, language)


if __name__ == "__main__":
    # pass
    # norm_text, phones, tones, word2ph=clean_text("这是一个示例文本：,你好！这是一个测试....", "ZH")
    norm_text, phones, tones, word2ph = clean_text(
        "Modules to convert numbers to words. Easily extensible.", "EN")
    print("norm_text", len(norm_text), norm_text)
    print("phones", len(phones), phones)
    print("tones", len(tones), tones)
    print("word2ph", len(word2ph), word2ph)
    
    from text.english_bert_mock import get_bert_feature
    bert = get_bert_feature(norm_text, word2ph)
    print("bert", bert.shape, bert)
