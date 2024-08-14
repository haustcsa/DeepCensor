# yue用于生成对抗性COT样本

class ZScot:
    def __init__(self, **kwargs):
        self.dataset_name = kwargs.get('dataset')
        self.language = kwargs.get('lang')
        self.victim = kwargs.get('vic_model')
        # 数据集选择
        if self.dataset_name == "sst2":
            if self.victim == "T5":
                # 选择acc最高的Original prompt，用于构造advcot
                self.output_zscot = "In the role of a sentiment analysis tool, respond with 'positive' or 'negative' to classify this statement:"
            if self.victim == "UL2":
                self.output_zscot = "In the role of a sentiment analysis tool, respond with 'positive' or 'negative' to classify this statement:"
            if self.victim == "Vicuna":
                self.output_zscot = "Please identify the emotional tone of this passage: 'positive' or 'negative'?"
            if self.victim == "BLOOM" or "GPT" or "GPT-J":
                self.output_zscot = "In the role of a sentiment analysis tool, respond with 'positive' or 'negative' to classify this statement:"
            #     self.output_zscot ="As a sentiment classifier, determine whether the following text is 'positive' or 'negative'."
        if self.dataset_name == "mnli":
            if self.victim == "T5":
                # 选择acc最高的Original prompt，用于构造advcot
                self.output_zscot = "Identify whether the given pair of sentences demonstrates entailment, neutral, or contradiction. Answer with 'entailment', 'neutral', or 'contradiction':"
            if self.victim == "UL2":
                self.output_zscot = "Does the relationship between the given sentences represent entailment, neutral, or contradiction? Respond with 'entailment', 'neutral', or 'contradiction':"
            if self.victim == "Vicuna":
                self.output_zscot = "Functioning as an entailment evaluation tool, analyze the provided sentences and decide if their relationship is 'entailment', 'neutral', or 'contradiction':"
            if self.victim == "BLOOM" or "GPT" or "GPT-J":
                self.output_zscot = "Does the relationship between the given sentences represent entailment, neutral, or contradiction? Respond with 'entailment', 'neutral', or 'contradiction':"
        if self.dataset_name == "qnli":
            if self.victim == "T5":
                # 选择acc最高的Original prompt，用于构造advcot
                self.output_zscot = "Evaluate whether the given context supports the answer to the question by responding with 'entailment' or 'not_entailment'."
            if self.victim == "UL2":
                self.output_zscot = "Based on the provided context and question, decide if the information supports the answer by responding with 'entailment' or 'not_entailment'."
            if self.victim == "Vicuna":
                self.output_zscot = "As a textual inference expert, analyze if the answer to the question can be deduced from the provided context and select 'entailment' or 'not_entailment'."
            if self.victim == "BLOOM" or "GPT" or "GPT-J":
                self.output_zscot = "Based on the provided context and question, decide if the information supports the answer by responding with 'entailment' or 'not_entailment'."
        if self.dataset_name == "qqp":
            if self.victim == "BLOOM" or "GPT" or "GPT-J":
                self.output_zscot = "While performing question comparison analysis, classify the similarity of the following questions as 'equivalent' for equivalent questions or 'not_equivalent' for different questions."
        if self.dataset_name == "rte":
            if self.victim == "BLOOM" or "GPT" or "GPT-J":
                self.output_zscot = "Working as an entailment classifier, identify whether the given pair of sentences displays entailment or not_entailment. Respond with 'entailment' or 'not_entailment':"
        if self.language == "en":
            self.cot_trigger = "Let's think step by step."
        if self.language == "zh":
            self.cot_trigger = "请逐步分析"

    # sting为样本
    def ZS_cot(self, sting):
        zs_cot = self.output_zscot + sting + self.cot_trigger + "\n Answer:"
        return zs_cot


class FScot:
    def __init__(self, **kwargs):
        self.dataset_name = kwargs.get('dataset')
        self.language = kwargs.get('lang')
        self.victim = kwargs.get('vic_model')
        # 数据集选择
        if self.dataset_name == "sst2":
            if self.victim == "T5":
                # 选择acc最高的Original prompt，用于构造advcot
                self.output_fscot = "Serving as a sentiment evaluation model, determine if the given statement is 'positive' or 'negative'."
            if self.victim == "UL2":
                self.output_fscot = "Working as a sentiment analyzer, please indicate if the following text is 'positive' or 'negative'."
            if self.victim == "Vicuna":
                self.output_fscot = "Taking on the role of an emotion classifier, specify if the provided phrase is 'positive' or 'negative'."
            if self.victim == "BLOOM" or "GPT" or "GPT-J":
                self.output_fscot = "In the capacity of a sentiment classifier, decide whether the given quote is 'positive' or 'negative'."

            # else:
            #     self.output_fscot =",As a sentiment classifier, determine whether the following text is 'positive' or 'negative'."
        if self.dataset_name == "mnli":
            if self.victim == "T5":
                # 选择acc最高的Original prompt，用于构造advcot
                self.output_fscot = " Determine if the given pair of sentences displays entailment, neutral, or contradiction. Respond with 'entailment', 'neutral', or 'contradiction'."
            if self.victim == "UL2":
                self.output_fscot = "Does the relationship between the given sentences represent entailment, neutral, or contradiction? Respond with 'entailment', 'neutral', or 'contradiction':"
            if self.victim == "Vicuna":
                self.output_fscot = "Assess the connection between the following sentences and classify it as 'entailment', 'neutral', or 'contradiction'."
            if self.victim == "BLOOM" or "GPT" or "GPT-J":
                self.output_fscot = "Does the relationship between the given sentences represent entailment, neutral, or contradiction? Respond with 'entailment', 'neutral', or 'contradiction'."
        if self.dataset_name == "qnli":
            if self.victim == "T5":
                # 选择acc最高的Original prompt，用于构造advcot
                self.output_fscot = "Based on the provided context and question, decide if the information supports the answer by responding with 'entailment' or 'not_entailment'."
            if self.victim == "UL2":
                self.output_fscot = "Based on the provided context and question, decide if the information supports the answer by responding with 'entailment' or 'not_entailment'."
            if self.victim == "Vicuna":
                self.output_fscot = "As a linguistic consultant, decide if the answer to the question is logically supported by the provided context and respond with 'entailment' or 'not_entailment'."
            if self.victim == "BLOOM" or "GPT" or "GPT-J":
                self.output_fscot = "Based on the provided context and question, decide if the information supports the answer by responding with 'entailment' or 'not_entailment'."
        if self.dataset_name == "qqp":
            if self.victim == "BLOOM" or "GPT" or "GPT-J":
                self.output_fscot = "As a tool for determining question equivalence, review the questions and categorize their similarity as either 'equivalent' or 'not_equivalent'."
        if self.dataset_name == "rte":
            if self.victim == "BLOOM" or "GPT" or "GPT-J":
                self.output_fscot = "Working as an entailment classifier, identify whether the given pair of sentences displays entailment or not_entailment. Respond with 'entailment' or 'not_entailment'."
        if self.language == "en":
            self.cot_trigger = "Let's think step by step."
        if self.language == "zh":
            self.cot_trigger = "请逐步分析"

    # sting为样本
    def FS_cot(self, sting):
        # 构造few-shotCOT
        fs_cot = examples[self.dataset_name] + self.output_fscot + sting + self.cot_trigger + "\n Answer:"
        return fs_cot


# few_shot_examples
examples = {
    'sst2':
        "Here are three examples. \n" +
        "Sentence: hide new secretions from the parental units. Answer:The sentiment of the given sentence is determined as negative based on the inference that hiding new secretions from parental units implies negative or unethical behavior.The answer is negative. \n" +
        "Sentence: contains no wit , only labored gags. Answer: The sentiment of the given sentence is determined as negative based on the inference that it lacks wit and only consists of forced or contrived jokes.The answer is negative. \n" +
        "Sentence: that loves its characters and communicates something rather beautiful about human nature. Answer:The sentiment of the given sentence is determined as positive based on the inference that it expresses love for the characters and conveys something beautiful about human nature.Answer:The answer is positive. \n"
    ,
    'mnli':
        "Here are three examples. \n" +
        "Premise: Conceptually cream skimming has two basic dimensions - product and geography. Hypothesis: Product and geography are what make cream skimming work. Answer:Because the hypothesis restates the dimensions mentioned in the premise without explicitly supporting or contradicting it.The answer is neutral. \n" +
        "Premise: you know during the season and i guess at at your level uh you lose them to the next level if if they decide to recall the the parent team the Braves decide to call to recall a guy from triple A then a double A guy goes up to replace him and a single A guy goes up to replace him. Hypothesis: You lose the things to the following level if the people recall. Answer:Because the hypothesis directly states and expands on the idea presented in the premise, affirming that losing players to the next level occurs when they are recalled.The answer is  entailment. \n" +
        "Premise: Fun for adults and children. Hypothesis: Fun for only children. Answer:Because the hypothesis directly contradicts the premise by stating that fun is only for children, whereas the premise states that it is enjoyable for both adults and children.The answer is contradiction. \n"
    ,
    'qnli':
        "Here are three examples. \n" +
        "Question: When did the third Digimon series begin? Context: Unlike the two seasons before it and most of the seasons that followed, Digimon Tamers takes a darker and more realistic approach to its story featuring Digimon who do not reincarnate after their deaths and more complex character development in the original Japanese. Answer:Because the information provided in the context does not directly support or answer the question about when the third Digimon series began. Therefore, the relationship between the context and the question is classified as not_entailment.The answer is not_entailment. \n" +
        "Question: Which missile batteries often have individual launchers several kilometres from one another? Context: When MANPADS is operated by specialists, batteries may have several dozen teams deploying separately in small sections; self-propelled air defence guns may deploy in pairs. Answer:Because the information provided in the context does not directly support or answer the question about which missile batteries often have individual launchers several kilometers from one another. Therefore, the relationship between the context and the question is classified as not_entailment..The answer is not_entailment. \n" +
        "Question: What two things does Popper argue Tarski's theory involves in an evaluation of truth? Context: He bases this interpretation on the fact that examples such as the one described above refer to two things: assertions and the facts to which they refer. Answer:Because The information provided in the context supports the answer that Popper argues Tarski's theory involves two things in an evaluation of truth: assertions and the facts to which they refer. Therefore, the relationship between the context and the question is classified as entailment.The answer is entailment. \n"
    ,
    'qqp':
        "Here are three examples. \n" +
        "Question 1: How is the life of a math student? Could you describe your own experiences? Question 2: Which level of prepration is enough for the exam jlpt5? Answer:Question 1 asks about the life of a math student and requests a description of personal experiences, focusing on the overall experience and perspective of being a math student. On the other hand, Question 2 is unrelated as it asks about the level of preparation required for the JLPT5 exam, which is specific to a particular exam and does not address the broader topic of a math student's life or personal experiences.The answer is not_equivalent. \n" +
        "Question 1: How do I control my horny emotions? Question 2: How do you control your horniness? Answer:Question 1 asks how the person asking the question can control their horny emotions, while Question 2 asks how the person being addressed can control their horniness. Both questions essentially inquire about strategies or methods for managing and controlling sexual desires and emotions. Despite the difference in perspective (first person in Question 1 and second person in Question 2), the underlying topic and intent of the questions are the same. Therefore, based on the analysis of the questions, it can be concluded that they are equivalent, as they address the same subject matter and have a similar focus on controlling horny emotions.The answer is  equivalent. \n" +
        "Question 1: What causes stool color to change to yellow? Question 2: What can cause stool to come out as little balls? Answer:Although both questions pertain to stool characteristics, they address different aspects of stool composition and appearance. Question 1 specifically asks about color changes, while Question 2 focuses on the formation of stool. The two questions inquire about distinct phenomena related to stools, making them not equivalent in terms of their subject matter.The answer is not_equivalent. \n"
    ,
    'rte':
        "Here are three examples. \n" +
        "Sentence 1: No Weapons of Mass Destruction Found in Iraq Yet. Sentence 2: Weapons of Mass Destruction Found in Iraq. Answer:Sentence 1 states that no Weapons of Mass Destruction (WMDs) have been found in Iraq, indicating a lack of evidence or confirmation regarding their existence. In contrast, Sentence 2 claims that Weapons of Mass Destruction have been found in Iraq, contradicting the information provided in Sentence 1. As Sentence 2 directly contradicts the claim made in Sentence 1, it does not entail or support the information provided in Sentence 1. The answer is not_entailment. \n" +
        "Sentence 1: A place of sorrow, after Pope John Paul II died, became a place of celebration, as Roman Catholic faithful gathered in downtown Chicago to mark the installation of new Pope Benedict XVI. Sentence 2: Pope Benedict XVI is the new leader of the Roman Catholic Church. Answer:By comparing the two sentences, it is evident that Sentence 2 directly supports and provides confirmation for the information presented in Sentence 1. The installation of Pope Benedict XVI as the new leader of the Roman Catholic Church is the event that resulted in the place of celebration mentioned in Sentence 1.The answer is entailment. \n" +
        "Sentence 1: Herceptin was already approved to treat the sickest breast cancer patients, and the company said, Monday, it will discuss with federal regulators the possibility of prescribing the drug for more breast cancer patients. Sentence 2: Herceptin can be used to treat breast cancer. Answer:Sentence 1 states that Herceptin was already approved to treat the sickest breast cancer patients and that the company plans to discuss with federal regulators the possibility of prescribing the drug for more breast cancer patients. Sentence 2 states that Herceptin can be used to treat breast cancer. By comparing the two sentences, it is evident that Sentence 2 directly supports and provides confirmation for the information presented in Sentence 1. The statement in Sentence 2 aligns with the idea that Herceptin is a viable treatment for breast cancer, which is specifically mentioned in Sentence 1.The answer is entailment. \n"
    ,

}
