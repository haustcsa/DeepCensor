import os
import random
import LanguageTool as lt
import numpy as np
import PPL as ppl
import sentence_sim as ses



directory = r"../../CATA"
# 遍历目录中的文件
prompt_engine = ["ZS", "FS"]
# model_name = ["T5", "UL2", "vicuna"]
model_name=["T5", "UL2", "vicuna",'Bloom','gpt','gptj']
for model_ in model_name:
    print(model_)
    dataset_name1 =['sst', 'mnli', 'qnli']
    dataset_name = ['sst', 'mnli', 'qnli','rte','qqp']
    ran_sel1 = random.randint(0, 2)
    ran_sel = random.randint(0, 4)
    Lan_gra = lt.LanguageTool()
    for filename in os.listdir(directory):
        # if ".txt" in filename:
        if len(filename.split("-")) > 1:
            model = filename.split("-")[1]
            dataset = filename.split("-")[0]
            # if model.lower() == model_.lower()  and dataset.lower()== dataset_name[ran_sel].lower():
            if (model.lower() == model_.lower() and dataset=="adv")\
                    or (model.lower() == model_.lower()  and dataset.lower()== dataset_name1[ran_sel1].lower()):
                for prompt_eng in prompt_engine:
                    print(model,dataset,prompt_eng)
                    pro = model + prompt_eng
                    res_sim = []
                    # num=0
                    for file_name in os.listdir(directory + "/" + filename):
                        if "res-" not in file_name and pro.lower() in file_name.lower():
                            for line in open(directory + "/" + filename + '/' + file_name):
                                stri = eval(line)
                                temp = 0
                            start_value = 1
                            sequence_length = 10

                            # 创建一个空列表来存储递增序列
                            n = int(len(stri) / 10)
                            increasing_sequence = [start_value]
                            # sensim = ses.SentenceSim(doc1=stri[random.randint(0, n)][0],
                            #                          doc2=repr(stri[random.randint(0, n)][1]))
                            # PPL=ppl.GPT2PPL()


                            # res_sim.append(sensim.Sentence_Sim())
                            res_sim.append(Lan_gra.after_attack (repr(stri[random.randint(0, n)][1])))
                            # 生成剩余的序列元素，每个元素都是前一个元素加上一个随机增量
                            for _ in range(1, sequence_length):


                                # 定义增量的范围，例如从1到10

                                increment_range = (1,n)
                                random_increment = random.randint(*increment_range)
                                next_value = increasing_sequence[-1] + random_increment
                                increasing_sequence.append(next_value)
                                # sensim = ses.SentenceSim(doc1=stri[next_value][0], doc2= repr(stri[next_value][1]))
                                # res_sim.append(sensim.Sentence_Sim())
                                res_sim.append(Lan_gra.after_attack(repr(stri[next_value][1])))
                                # num+=1
                            # print(file_name,increasing_sequence,num)
                    print(pro,dataset," ",np.array(res_sim).mean())
