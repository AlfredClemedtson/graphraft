import torch
from transformers import LogitsProcessor, LlamaTokenizer

class MyLogitsProcessor(LogitsProcessor):
    def __init__(self, allowed_queries_ids, starting_ids):
        self.allowed_queries_ids = allowed_queries_ids
        self.starting_ids = starting_ids
        self.previous_queries_ids = set()

    def __call__(self, input_ids, scores):
        new_scores = torch.zeros_like(scores) + float("-inf")
        for i in range(len(input_ids)):
            gen_ids = self.take_generated_ids(input_ids[i,...])
            allowed_ids = self.allowed_next_ids(gen_ids)
            new_scores[i, allowed_ids] = scores[i, allowed_ids]
        #     print(tokenizer.decode(gen_ids), [tokenizer.decode(allowed_id) for allowed_id in allowed_ids])
        # print()
        return new_scores

    def allowed_next_ids(self, gen_ids):
        return list({query_ids[0, gen_ids.shape[0]].item() for query_ids in self.allowed_queries_ids if
                 query_ids.shape[1] > gen_ids.shape[0] and torch.equal(query_ids[0, :gen_ids.shape[0]], gen_ids)})

    def take_generated_ids(self, full_text_ids):
        l = self.starting_ids.shape[0]
        for i in range(full_text_ids.shape[0]):
            if torch.equal(full_text_ids[i:i+l], self.starting_ids):
                return full_text_ids[i+l:]
        raise Exception("Generate pattern string prompt thing is missing :/")


class MyLogitsProcessorList(list):
    def __init__(self, allowed_queries_ids, starting_ids):
        super().__init__()
        self.logits_processor = MyLogitsProcessor(allowed_queries_ids, starting_ids)

    def __iter__(self):
        for _ in range(1):
            yield self.logits_processor

    def __len__(self):
        return 1