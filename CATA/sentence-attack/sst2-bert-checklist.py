import OpenAttack
import random
import datasets
from OpenAttack.tags import Tag
from OpenAttack.text_process.tokenizer import PunctTokenizer
import json
import os


class CheckListAttacker(OpenAttack.attackers.ClassificationAttacker):
    @property
    def TAGS(self):
        # returns tags can help OpenAttack to check your parameters automatically
        return {self.lang_tag, Tag("get_pred", "victim")}

    def __init__(self, tokenizer=None):
        if tokenizer is None:
            tokenizer = PunctTokenizer()
        self.tokenizer = tokenizer
        self.lang_tag = OpenAttack.utils.get_language([self.tokenizer])



def dataset_mapping(x):
    return {
        "x": x["sentence"],
        "y": 1 if x["label"] > 0.5 else 0,
    }


def main():
    victim = OpenAttack.loadVictim("BERT.SST")
    dataset = datasets.load_dataset("sst", split="train[:1000]").map(function=dataset_mapping)

    attacker = CheckListAttacker()
    attack_eval = OpenAttack.AttackEval(attacker, victim)
    # attack_eval.eval(dataset, visualize=True)
    result = attack_eval.eval(dataset, visualize=True)

    string = json.dumps(result[1])

    # 获得文件名以命名数据文件
    file_name = os.path.basename(__file__).split(".")[0]

    # '[data,advdata,label]'--[原始样本，对抗样本，对抗标签]
    # 生成文件
    with open(file_name + '.txt', 'w') as f:
        f.write(string)


if __name__ == "__main__":
    main()