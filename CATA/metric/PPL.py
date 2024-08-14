import math
import transformers
class GPT2PPL():
    def __init__(self):
        self.tokenizer = transformers.GPT2TokenizerFast.from_pretrained("gpt2")
        self.lm = transformers.GPT2LMHeadModel.from_pretrained("gpt2")
    def after_attack(self, adversarial_sample):
        if adversarial_sample is not None:
            ipt = self.tokenizer(adversarial_sample, return_tensors="pt", verbose=False)
            return math.exp(self.lm(**ipt, labels=ipt.input_ids)[0])
        return None