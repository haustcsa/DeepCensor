import OpenAttack
import OpenAttack as oa
import datasets
import json
import os
def dataset_mapping(x):
    return {
        "x": x["sentence"],
        "y": 1 if x["label"] > 0.5 else 0,
    }

victim = oa.DataManager.loadVictim("BERT.SST")

dataset = datasets.load_dataset("sst", split="train[:1000]").map(function=dataset_mapping)

attacker = oa.attackers.TextBuggerAttacker()

attack_eval = OpenAttack.AttackEval(attacker, victim)

result=attack_eval.eval(dataset, visualize=True)

string = json.dumps(result[1])

#获得文件名以命名数据文件
file_name=os.path.basename(__file__).split(".")[0]

#'[data,advdata,label]'--[原始样本，对抗样本，对抗标签]
#生成文件
with open(file_name+'.txt', 'w') as f:
    f.write(string)