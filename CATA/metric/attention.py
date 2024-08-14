import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM


class Visualizer:
    def __init__(self, model) -> None:
        self.tokenizer = AutoTokenizer.from_pretrained(model)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model)
    def _map_subwords_to_words(self, sentence):
        tokens = self.tokenizer.tokenize(sentence)
        mapping = []
        word_idx = 0
        for token in tokens:
            if token.startswith("▁"):
                mapping.append(word_idx)
                word_idx += 1
            else:
                mapping.append(word_idx - 1)
        return mapping, tokens

    def _normalize_importance(self, word_importance):

        min_importance = np.min(word_importance)
        max_importance = np.max(word_importance)
        return (word_importance - min_importance) / (max_importance - min_importance)

    def vis_by_grad(self, input_sentence, label) -> dict:

        self.model.eval()

        mapping, tokens = self._map_subwords_to_words(input_sentence)
        words = "".join(tokens).replace("▁", " ").split()

        inputs = self.tokenizer(input_sentence, return_tensors="pt")
        embeddings = self.model.get_input_embeddings()(inputs['input_ids'])
        embeddings.requires_grad_()
        embeddings.retain_grad()

        labels = self.tokenizer(label, return_tensors="pt")["input_ids"]
        outputs = self.model(inputs_embeds=embeddings, attention_mask=inputs['attention_mask'], labels=labels)
        outputs.loss.backward()

        grads = embeddings.grad
        word_grads = [torch.zeros_like(grads[0][0]) for _ in range(len(words))]  # Initialize gradient vectors for each word

        # Aggregate gradients for each word
        for idx, grad in enumerate(grads[0][:len(mapping)]):
            word_grads[mapping[idx]] += grad

        words_importance = [grad.norm().item() for grad in word_grads]
        normalized_importance = self._normalize_importance(words_importance)

        return dict(zip(words, normalized_importance))
input_sentence=''
str="entailment"
Vis=Visualizer("google/flan-t5-large")

print(Vis.vis_by_grad(input_sentence,str))