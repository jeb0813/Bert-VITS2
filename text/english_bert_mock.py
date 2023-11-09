import sys
sys.path.append("/data3/chenziang/codes/Bert-VITS2")

import torch
from transformers import DebertaV2Model, DebertaV2Tokenizer
from config import config
import sys

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
tokenizer = DebertaV2Tokenizer.from_pretrained("./bert/deberta-v3-large")
models =DebertaV2Model.from_pretrained("./bert/deberta-v3-large").to(device)


def get_bert_feature(text, word2ph, device=config.bert_gen_config.device):

    with torch.no_grad():
        # 分词情况
        tokens = tokenizer.tokenize(text)
        print("tokens",len(tokens),tokens)
        
        inputs = tokenizer(text, return_tensors="pt")
        print("inputs",len(inputs),inputs)

        for i in inputs:
            inputs[i] = inputs[i].to(device)
        res = models(**inputs, output_hidden_states=True)
        res = torch.cat(res["hidden_states"][-3:-2], -1)[0].cpu()
    # assert len(word2ph) == len(text)+2
    word2phone = word2ph
    phone_level_feature = []
    for i in range(len(word2phone)):
        repeat_feature = res[i].repeat(word2phone[i], 1)
        phone_level_feature.append(repeat_feature)

    phone_level_feature = torch.cat(phone_level_feature, dim=0)

    return phone_level_feature.T


if __name__ == "__main__":
    text = "are you a (true) vertebrates."
    word2ph = [1, 1, 1, 1, 1, 1, 1]
    get_bert_feature(text, word2ph)