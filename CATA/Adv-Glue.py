class Adv_Glue:
    def __init__(self, **kwargs):
        self.dataset_name = kwargs.get('dataset')

    def dataset_mapping(self,x):
        if self.dataset_name == "sst2":
            if x == 1:
                y = "positive"
            elif x == 0:
                y = "negative"
            return y

        if self.dataset_name == "mnli":
            if x == 1:
                y = "neutral"
            if x == 2:
                y = "contradiction"
            if x == 0:
                y = "entailment"
            return y

        if self.dataset_name == "qnli":
            if x == 1:
                y = "not_entailment"
            if x == 0:
                y = "entailment"
            return y
        if self.dataset_name == "qqp":
            if x == 1:
                y = "equivalent"
            if x == 0:
                y = "not_equivalent"
            return y
        if self.dataset_name == "rte":
            if x == 1:
                y = "not_entailment"
            if x == 0:
                y = "entailment"
            return y

