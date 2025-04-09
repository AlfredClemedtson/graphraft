from transformers import GenerationConfig

from constrained_decoding.logits_processor import MyLogitsProcessor

class SequenceRanker:
    def __init__(self, model, tokenizer, device, start_of_generation_tokens: str, end_of_generation_token: str):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.start_of_generation_tokens = start_of_generation_tokens
        self.end_of_generation_token = end_of_generation_token
        self.starting_ids = tokenizer(start_of_generation_tokens, add_special_tokens=False, return_tensors="pt").input_ids[0].to(device)
        self.end_token_id = tokenizer(end_of_generation_token, add_special_tokens=False, return_tensors="pt").input_ids[0].to(device)

    def rank_sequences(self, prompt: str, possible_sequences: str, max_beam_width: int) -> list[str]:
        beam_width = min(max_beam_width, len(possible_sequences))
        max_new_tokens = max([len(self.tokenizer.tokenize(sequence))+10 for sequence in possible_sequences])
        generation_config = GenerationConfig(num_return_sequences=beam_width, num_beams=beam_width, #early_stopping=True
                                             do_sample=False, max_new_tokens=max_new_tokens,)
        allowed_sequences_ids = [
            self.tokenizer(query + self.end_of_generation_token, add_special_tokens=False, return_tensors="pt").input_ids.to(self.device)
            for query in possible_sequences]

        logits_processor = MyLogitsProcessor(allowed_sequences_ids, self.starting_ids, self.end_token_id)

        tokenized_prompt = self.tokenizer(prompt ,return_tensors="pt").to(self.device)
        outputs = self.model.generate(**tokenized_prompt,
                                 generation_config=generation_config,
                                 logits_processor=[logits_processor],
                                 )
        decoded_outputs = self.tokenizer.batch_decode(outputs, skip_special_tokens=False)
        top_sequences = [text.split(self.start_of_generation_tokens)[-1].split(self.end_of_generation_token)[0]
                         for text in decoded_outputs]
        return top_sequences