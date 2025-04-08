import torch
from transformers import LogitsProcessor, LlamaTokenizer

class MyLogitsProcessor(LogitsProcessor):
    def __init__(self, allowed_queries_ids, starting_ids, end_token_id, tokenizer = None):
        self.allowed_queries_ids = allowed_queries_ids
        self.starting_ids = starting_ids
        self.end_token_id = end_token_id
        self.previous_queries_ids = set()
        self.tokenizer = tokenizer

    def __call__(self, input_ids, scores):
        new_scores = torch.zeros_like(scores) + float("-inf")
        for i in range(len(input_ids)):
            gen_ids = self.take_generated_ids(input_ids[i,...])
            allowed_ids = self.allowed_next_ids(gen_ids)
            new_scores[i, allowed_ids] = scores[i, allowed_ids]
            if self.tokenizer is not None: # If tokenizer is given, print sequences during generation.
                print(self.tokenizer.decode(gen_ids), [self.tokenizer.decode(id) for id in allowed_ids])
        if self.tokenizer is not None:
            print()
        return new_scores

    def allowed_next_ids(self, gen_ids):
        if gen_ids.shape[0] > 0 and gen_ids[-1] == self.end_token_id:
            return [self.end_token_id]
        else:
            return list({query_ids[0, gen_ids.shape[0]].item() for query_ids in self.allowed_queries_ids if
                 query_ids.shape[1] > gen_ids.shape[0] and torch.equal(query_ids[0, :gen_ids.shape[0]], gen_ids)})

    def take_generated_ids(self, full_text_ids):
        l = self.starting_ids.shape[0]
        for i in range(full_text_ids.shape[0]):
            if torch.equal(full_text_ids[i:i+l], self.starting_ids):
                return full_text_ids[i+l:]
        raise Exception("Generate pattern string prompt thing is missing :/")