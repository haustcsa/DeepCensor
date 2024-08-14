import os
import random
from OpenAttack.tags import Tag
from OpenAttack.text_process.tokenizer import PunctTokenizer
import OpenAttack
import transformers
import datasets



class NLIWrapper(OpenAttack.classifiers.Classifier):
    def __init__(self, model: OpenAttack.classifiers.Classifier):
        self.model = model

    def get_pred(self, input_):
        return self.get_prob(input_).argmax(axis=1)

    def get_prob(self, input_):
        ref = self.context.input["question"]
        input_sents = [ref + "</s></s>" +sent  for sent in input_]
        # print(input_sents)
        return self.model.get_prob(
            input_sents
        )

def dataset_mapping(x):
    # print(x)
    return {
        "x": x["sentence"],
        "y": x["label"],
        "question": x["question"]
    }


def main():
    print("Load model")
    tokenizer = transformers.AutoTokenizer.from_pretrained("ModelTC/bert-base-uncased-qnli")
    model = transformers.AutoModelForSequenceClassification.from_pretrained("ModelTC/bert-base-uncased-qnli",output_hidden_states=False)
    victim = OpenAttack.classifiers.TransformersClassifier(model, tokenizer, model.bert.embeddings.word_embeddings)
    victim = NLIWrapper(victim)

    print("New Attacker")
    attacker = StressTestAttacker()

    dataset = datasets.load_dataset("glue", "qnli", split="train[:1000]").map(function=dataset_mapping)

    print("Start attack")

    attack_eval = OpenAttack.AttackEval(attacker, victim, metrics=[
        OpenAttack.metric.EditDistance(),
        OpenAttack.metric.ModificationRate()
    ])
    result = attack_eval.eval(dataset, visualize=True)
    # print("----------------------",result)
    import json
    # 转化为json文件输出
    # 获得文件名以命名数据文件
    file_name = os.path.basename(__file__).split(".")[0]
    string = json.dumps(result[1])
    with open(file_name + '.txt', 'w') as f:
        f.write(string)


if __name__ == "__main__":
    main()

