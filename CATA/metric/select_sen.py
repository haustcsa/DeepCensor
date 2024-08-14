import os
import pandas as pd
import attention as att

directory = r"../mnli-T5"
prompt_engine = ["ZS", "FS"]
attack_type=["deepwordbug"]

# attack_type=["TextBugger","BertAttack","CheckList"]
# pe=prompt_engine[0]
# attack=attack_type[0]
Vis=att.Visualizer("google/flan-t5-large")
for pe in prompt_engine:
    for attack in attack_type:
        for filename in os.listdir(directory):
            flag = []
            str=''
            #select attack type and prompt engineer
            if  "res-" in filename and \
                    attack.lower() in filename.lower()\
                    and pe in filename :
                dir_path = directory + "\\" + filename
                df = pd.read_csv(dir_path, index_col=None, header=None)
                for i in range(len(df[0])):
                    if df[0][i] !=df[1][i] and df[2][i] ==df[1][i]:
                        flag.append(i)

                dir_path_out_res=dir_path.replace("res-","")
                for line in open(dir_path_out_res):
                    sting = eval(line)
                print(dir_path,"\n",dir_path_out_res)
                for j in flag:
                    print(j,df[2][j])
                    print("原始样本：",sting[j][0])
                    # print(Vis.vis_by_grad(sting[j][0], str))
                    print(df[0][j])
                    print("对抗样本",repr(sting[j][1]))
                    # print(Vis.vis_by_grad(repr(sting[j][1]), str))

                    break
                break