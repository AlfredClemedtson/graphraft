import time
import os
import argparse
import numpy as np
import torch
from transformers import BitsAndBytesConfig, GenerationConfig, AutoModelForCausalLM, PreTrainedTokenizerBase, \
    AutoTokenizer
from datasets import Dataset
from datasets import load_from_disk
from peft import LoraConfig, PeftModel, get_peft_model
from trl import SFTConfig, SFTTrainer

from sequence_ranker import SequenceRanker

PAD_TOKEN = "<pad>"
START_OF_GENERATION_TOKENS = "<start_of_turn>assistant\n"
END_OF_GENERATION_TOKEN = "<eos>"  # "<end_of_turn>"

class LLM1:
    def __init__(self, device, model_dir, adapter_dir=None, beam_width=5):
        # Model
        bnb_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_use_double_quant=True, bnb_4bit_quant_type="nf4",
                                        bnb_4bit_compute_dtype=torch.bfloat16,)
        self.tokenizer = AutoTokenizer.from_pretrained(model_dir)
        self.tokenizer.padding_side = "right"

        model = AutoModelForCausalLM.from_pretrained(
            model_dir,
            quantization_config=bnb_config if device != torch.device('cpu') else None,
            torch_dtype=torch.bfloat16,
            attn_implementation="eager",
            low_cpu_mem_usage=True,
        ).to(device)
        if adapter_dir is None:
            lora_config = LoraConfig(r=64, lora_alpha=64, target_modules=None, lora_dropout=0.05, bias="none",
                                     task_type="CAUSAL_LM", )
            self.model = get_peft_model(model=model, peft_config=lora_config)
        else:
            self.model = PeftModel.from_pretrained(model, adapter_dir)

        self.sequence_ranker = SequenceRanker(model=self.model, tokenizer=self.tokenizer, device=device,
                                              start_of_generation_tokens=START_OF_GENERATION_TOKENS,
                                              end_of_generation_token=END_OF_GENERATION_TOKEN)
        self.beam_width = beam_width


    def put_in_inference_mode(self):
        self.tokenizer.padding_side = 'left'
        self.tokenizer.pad_token = self.tokenizer.eos_token
        # Todo?
        # Put into inference mode? No grad, padding side: left?
        # raise NotImplemented


    @staticmethod
    def sort_cyphers(data: dict) -> dict:
        cyphers, hits, num_results = data['cypher_queries'], data['hits'], data['num_results']
        data['cypher_queries'], data['hits'], data['num_results'] = zip(
            *sorted(zip(cyphers, hits, num_results), key=lambda x: (-x[1], x[2])))
        return data


    @staticmethod
    def best_label_is_good(data: dict, lowest_recall=1, lowest_precision=.1) -> bool:
        sorted_data = LLM1.sort_cyphers(data)
        precision = sorted_data['hits'][0] / sorted_data['num_results'][0]
        recall = sorted_data['hits'][0] / len(sorted_data['answer_ids'])
        return recall >= lowest_recall and precision >= lowest_precision


    @staticmethod
    def formatting_func(question):
        prompt = f"<start_of_turn>user\n{question}<end_of_turn>" + START_OF_GENERATION_TOKENS
        return prompt


    @staticmethod
    def format_and_add_true_label(question, data: dict) -> str:  # Not used yet
        prompt = LLM1.formatting_func(question)
        answer = LLM1.sort_cyphers(data)['cypher_queries'][0]
        prompt += answer + END_OF_GENERATION_TOKEN
        return prompt


    def predict_top_queries(self, question: str, possible_queries: list[str]):
        prompt = LLM1.formatting_func(question=question)
        top_queries = self.sequence_ranker.rank_sequences(prompt, possible_sequences=possible_queries,
                                                          max_beam_width=self.beam_width)
        return top_queries

    def train(self, train_dataset: Dataset, eval_dataset: Dataset, model_save_dir: str):
        # Load data
        max_seq_len = max([len(self.tokenizer.encode(x['text'])) for x in train_dataset]) + 10

        sft_config = SFTConfig(auto_find_batch_size=True,
                               gradient_accumulation_steps=1, #8
                               dataset_num_proc=8,
                               num_train_epochs=1,
                               learning_rate=2e-5,
                               optim="paged_adamw_8bit",
                               max_seq_length=max_seq_len,
                               eval_strategy="epoch",
                               save_strategy="epoch",
                               logging_steps=10,
                               output_dir=model_save_dir,
                               load_best_model_at_end=True,
                               )

        trainer = SFTTrainer(model=self.model,
                             args=sft_config,
                             processing_class=self.tokenizer,
                             train_dataset=train_dataset,
                             eval_dataset=eval_dataset,
                             # formatting_func=None #format here instead of mapping all in beforehand
                             )
        trainer.train()  # Train on response only is worse?


    def generate(self, dataset_dir: str, ratio: float, output_save_dir: str = None):
        qa_with_cyphers = load_from_disk(dataset_dir)
        qa_with_gen_cyphers = qa_with_cyphers \
            .filter(lambda _, i: i < int(len(qa_with_cyphers) * ratio), with_indices=True) \
            .map(lambda x: x | {'top_cypher_queries': self.predict_top_queries(x['question'], x['cypher_queries'])})
        if output_save_dir is not None:
            qa_with_gen_cyphers.save_to_disk(output_save_dir)
        return qa_with_gen_cyphers


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default=None)

    parser.add_argument('--train', action='store_true') #Should be false by default, true if given
    parser.add_argument('--generate_valid', action='store_true')
    parser.add_argument('--generate_test', action='store_true')

    parser.add_argument('--train_rate', type=float, default=1)
    parser.add_argument('--eval_rate', type=float, default=1)
    parser.add_argument('--valid_rate', type=float, default=1)
    parser.add_argument('--test_rate', type=float, default=1)

    parser.add_argument('--model_dir', type=str, default="neo4j/text2cypher-gemma-2-9b-it-finetuned-2024v1")  # use -Instruct?
    parser.add_argument('--adapter_dir', type=str, default=None)
    parser.add_argument('--model_save_name', type=str, default=None)
    parser.add_argument('--generate_save_dir', type=str, default=None)
    parser.add_argument('--beam_width', type=int, default=5)

    args = parser.parse_args()

    model_save_name = args.model_save_name if args.model_save_name is not None else f"llm1-{int(time.time()) % 100_000}"
    model_save_dir = os.path.join(f"{args.dataset}-models", model_save_name)

    if args.dataset not in ['prime', 'mag']:
        raise ValueError(f"Unknown dataset: {args.dataset}, must be either 'prime' or 'mag'")

    train_data_dir = f"{args.dataset}-data/qa_with_cypher_queries/train"
    valid_data_dir = f"{args.dataset}-data/qa_with_cypher_queries/valid"
    test_data_dir = f"{args.dataset}-data/qa_with_cypher_queries/test"

    generate_save_dir = f"{args.dataset}-data/qa_with_generated_cypher_queries" if args.generate_save_dir is None else args.generate_save_dir

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    llm1 = LLM1(device=device, model_dir=args.model_dir, adapter_dir=args.adapter_dir,
                beam_width=args.beam_width)

    if args.train:
        # Put in train mode
        max_train_datas = int(len(load_from_disk(train_data_dir)) * args.train_rate)
        qa_with_supervised_prompts_train = load_from_disk(train_data_dir)\
            .filter(lambda x: llm1.best_label_is_good(x, lowest_recall=1, lowest_precision=.1))\
            .filter(lambda _, i: i < max_train_datas, with_indices=True) \
            .map(lambda x: x | {'text': llm1.format_and_add_true_label(x['question'], x)})
        max_eval_datas = int(len(load_from_disk(valid_data_dir)) * args.eval_rate)
        qa_with_supervised_prompts_valid = load_from_disk(valid_data_dir) \
            .filter(lambda x: llm1.best_label_is_good(x, lowest_recall=1, lowest_precision=.1)) \
            .filter(lambda _, i: i < max_eval_datas, with_indices=True) \
            .map(lambda x: x | {'text': llm1.format_and_add_true_label(x['question'], x)})

        llm1.train(train_dataset=qa_with_supervised_prompts_train,
              eval_dataset=qa_with_supervised_prompts_valid, model_save_dir=model_save_dir)

    llm1.put_in_inference_mode()

    if args.generate_valid:
        valid_output_save_dir = os.path.join(generate_save_dir, "valid")
        _ = llm1.generate(dataset_dir=valid_data_dir, ratio=args.valid_rate, output_save_dir=valid_output_save_dir)

    if args.generate_test:
        test_output_save_dir = os.path.join(generate_save_dir, "test")
        _ = llm1.generate(dataset_dir=test_data_dir, ratio=args.test_rate, output_save_dir=test_output_save_dir)


if __name__ == '__main__':
    main()
