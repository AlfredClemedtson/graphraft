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

# PAD_TOKEN = ... #? what's best?
START_OF_GENERATION_TOKENS = "<start_of_turn>assistant\n"
END_OF_GENERATION_TOKEN = "<eos>"  # "<end_of_turn>"

class LLM1:
    def __init__(self, device, model_dir, adapter_dir = None, beam_width=5):
        # Model
        bnb_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_use_double_quant=True, bnb_4bit_quant_type="nf4",
                                        bnb_4bit_compute_dtype=torch.bfloat16,)
        self.tokenizer = AutoTokenizer.from_pretrained(model_dir)
        self.tokenizer.padding_side = 'right'

        model = AutoModelForCausalLM.from_pretrained(
            model_dir,
            quantization_config=bnb_config if device != torch.device('cpu') else None,
            torch_dtype=torch.bfloat16,
            attn_implementation="eager",
            low_cpu_mem_usage=True,
        ).to(device)
        # model.padding_side = 'right'
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
        # Todo
        raise NotImplemented


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


    def predict_top_queries(self, question: str, possible_queries: list[str],):
        prompt = LLM1.formatting_func(question=question)
        top_queries = self.sequence_ranker.rank_sequences(prompt, possible_sequences=possible_queries,
                                                          max_beam_width=self.beam_width)
        return top_queries

    # def add_predicted_cypher(data: dict, beam_width: int, sequence_ranker: SequenceRanker, print_options=[],) -> dict:
    #     prompt = self.formatting_func(question=data['question'], data=data, add_label=False)
    #     possible_sequences = data['cypher_queries']
    #     data['top_cypher_queries'] = sequence_ranker.rank_sequences(prompt=prompt, possible_sequences=possible_sequences, max_beam_width=beam_width)
    #
    #     if 'summary' in print_options:
    #         print(f"Top cyphers: {data['top_cypher_queries']}\n"
    #               f"Total #cyphers:     {len(data['cypher_queries'])}\n"
    #               f"Generated #cyphers: {len(data['top_cypher_queries'])}\n"
    #               f"Gen valid #cyphers: {len(set(data['top_cypher_queries']).intersection(data['cypher_queries']))}\n")
    #
    #     if 'add_details' in print_options:
    #         sorted_data = sort_cyphers(data)
    #         max_recall = 0
    #         print(f"Question: {data['question']}")
    #         for top_cypher in data['top_cypher_queries']:
    #             try:
    #                 i = sorted_data['cypher_queries'].index(top_cypher)
    #                 precision = sorted_data['hits'][i] / sorted_data['num_results'][i]
    #                 recall = sorted_data['hits'][i] / len(data['answer_ids'])
    #                 if 'predicted_recall_at_1' not in data.keys():
    #                     data['rank@1'] = i + 1
    #                     data['num_nodes_at_1'] = sorted_data['num_results'][i]
    #                     data['predicted_recall_at_1'] = recall
    #                 max_recall = max(max_recall, recall)
    #                 print(f"Rank: {i + 1}   Precision: {precision:.3f}   Recall: {recall:.3f}   Cypher query: {top_cypher}")
    #             except ValueError:
    #                 print("not expected...")
    #
    #         print(f"Max recall: {max_recall}, Recall@1: {data.get('predicted_recall_at_1', '-')}, "
    #               f"#nodes@1: {data.get('num_nodes_at_1', '-')} Rank@1: {data.get('rank@1', '-')}\n")
    #         data['predicted_max_recall'] = max_recall
    #
    #     if 'gpu_info' in print_options:
    #         print("torch.cuda.memory_allocated: %fGB" % (torch.cuda.memory_allocated(0) / 1024 / 1024 / 1024))
    #         print("torch.cuda.memory_reserved: %fGB" % (torch.cuda.memory_reserved(0) / 1024 / 1024 / 1024))
    #         print("torch.cuda.max_memory_reserved: %fGB" % (torch.cuda.max_memory_reserved(0) / 1024 / 1024 / 1024))
    #
    #     return data


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
    parser.add_argument('--test_rate', type=float, default=1)

    parser.add_argument('--model_dir', type=str, default="meta-llama/Llama-3.1-8B-Instruct")  # use -Instruct?
    parser.add_argument('--adapter_dir', type=str, default=None)
    parser.add_argument('--model_save_name', type=str, default=None)

    parser.add_argument('--beam_width', type=int, default=5)

    parser.add_argument('--output_valid_save_dir', type=str, default=None)
    parser.add_argument('--output_test_save_dir', type=str, default=None)

    args = parser.parse_args()

    # parser = argparse.ArgumentParser()
    # parser.add_argument('--train', action='store_true')
    # parser.add_argument('--evaluate', action='store_true')
    # parser.add_argument('--generate', action='store_true')
    # parser.add_argument('--model_dir', type=str,
    #                     default="neo4j/text2cypher-gemma-2-9b-it-finetuned-2024v1")
    # parser.add_argument('--dataset', type=str, default=None)
    # parser.add_argument('--adapter_dir', type=str, default=None)
    # parser.add_argument('--model_save_dir', type=str, default=None)
    # parser.add_argument('--data_dir', type=str, default=None)
    # parser.add_argument('--train_data_dir', type=str, default=None)
    # parser.add_argument('--valid_data_dir', type=str, default=None)
    # parser.add_argument('--gen_data_dir', type=str, default=None)
    # parser.add_argument('--eval_save_dir', type=str, default=None)
    # parser.add_argument('--gen_save_dir', type=str, default=None)
    # # parser.add_argument('--eval_save_dir', type=str, default="prime-data/qa_with_eval_cyphers")
    # # parser.add_argument('--gen_save_dir', type=str, default="prime-data/qa_with_gen_cyphers")
    # parser.add_argument('--use_base_prompt', action='store_true')
    # parser.add_argument('--beam_width', type=int, default=5)
    # parser.add_argument('--eval_fraction', type=float, default=1)
    # parser.add_argument('--gen_fraction', type=float, default=1)
    # args = parser.parse_args()
    #
    # do_train = args.train
    # do_evaluate = args.evaluate
    # do_generate = args.generate

    model_save_name = args.model_save_name if args.model_save_name is not None else f"llm1-{int(time.time()) % 100_000}"
    model_save_dir = os.path.join(f"{args.dataset}-models", model_save_name)

    if args.dataset not in ['prime', 'mag']:
        raise ValueError(f"Unknown dataset: {args.dataset}, must be either 'prime' or 'mag'")

    train_data_dir = os.path.join(f"{args.dataset}-data/qa_with_ner", 'train')
    valid_data_dir = os.path.join(f"{args.dataset}-data/qa_with_ner", 'valid')
    test_data_dir = os.path.join(f"{args.dataset}-data/qa_with_ner", 'test')

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # # Model
    # bnb_config = BitsAndBytesConfig(
    #     load_in_4bit=True,
    #     bnb_4bit_use_double_quant=True,
    #     bnb_4bit_quant_type="nf4",
    #     bnb_4bit_compute_dtype=torch.bfloat16,
    # )
    # tokenizer = AutoTokenizer.from_pretrained(model_dir)
    # tokenizer.padding_side = 'right'
    #
    # model = AutoModelForCausalLM.from_pretrained(
    #     model_dir,
    #     quantization_config=bnb_config if device != torch.device('cpu') else None,
    #     torch_dtype=torch.bfloat16,
    #     attn_implementation="eager",
    #     low_cpu_mem_usage=True,
    # ).to(device)
    # model.padding_side = 'right'

    # if adapter_dir is None:
    #     lora_config = LoraConfig(r=64, lora_alpha=64, target_modules=None, lora_dropout=0.05, bias="none", task_type="CAUSAL_LM", )
    #     model = get_peft_model(model=model, peft_config=lora_config)
    # else:
    #     model = PeftModel.from_pretrained(model, adapter_dir)
    llm1 = LLM1(device=device, model_dir=args.model_dir, adapter_dir=args.adapter_dir,
                beam_width=args.beam_width)

    if args.train:
        # Put in train mode
        qa_with_supervised_prompts_train = load_from_disk(train_data_dir)\
            .filter(lambda x: llm1.best_label_is_good(x, lowest_recall=1, lowest_precision=.1))\
            .map(lambda x: x | {'text': llm1.format_and_add_true_label(x['question'], x)})
        qa_with_supervised_prompts_valid = load_from_disk(valid_data_dir) \
            .filter(lambda x: llm1.best_label_is_good(x, lowest_recall=1, lowest_precision=.1)) \
            .map(lambda x: x | {'text': llm1.format_and_add_true_label(x['question'], x)})

        llm1.train(train_dataset=qa_with_supervised_prompts_train,
              eval_dataset=qa_with_supervised_prompts_valid, model_save_dir=model_save_dir)

    # Put into inference mode? No grad, padding side: left?
    llm1.put_in_inference_mode()

    if args.generate_valid:
        _ = llm1.generate(dataset_dir=valid_data_dir, ratio=args.valid_fraction, output_save_dir=eval_save_dir)

    if args.generate_test:
        _ = llm1.generate(dataset_dir=test_data_dir, ratio=args.test_fraction, output_save_dir=test_save_dir)

    # if do_evaluate:
    #     qa_with_cyphers = load_from_disk(valid_data_dir)
    #     qa_with_evaluation_result = qa_with_cyphers\
    #         .filter(lambda _, i: i < int(len(qa_with_cyphers) * eval_fraction), with_indices=True) \
    #         .map(lambda x: x | {'top_cypher_queries' : llm1.predict_top_queries(x['question'], x['cypher_queries'])})
    #         # .map(lambda data: add_predicted_cypher(data, sequence_ranker=sequence_ranker, beam_width=beam_width,
    #         #                                        print_options=['add_details', 'gpu_info']))
    #
    #     print(f"Avg recall@1: {np.mean(qa_with_evaluation_result['predicted_recall_at_1']):.2f}    "
    #           f"Avg top recall of 5: {np.mean(qa_with_evaluation_result['predicted_max_recall'])}    "
    #           f"Avg #nodes@1 {np.mean(qa_with_evaluation_result['num_nodes_at_1'])}")
    #     qa_with_evaluation_result.save_to_disk(eval_save_dir)
    #
    # if do_generate:
    #     # Generate cypher queries for all questions
    #     qa_with_cyphers = load_from_disk(gen_data_dir)
    #     qa_with_gen_cyphers = qa_with_cyphers \
    #         .filter(lambda _, i: i < int(len(qa_with_cyphers) * gen_fraction), with_indices=True) \
    #         .map(lambda x: x | {'top_cypher_queries': llm1.predict_top_queries(x['question'], x['cypher_queries'])})
    #         # .map(lambda data: add_predicted_cypher(data, sequence_ranker=sequence_ranker, beam_width=beam_width,))
    #                                                #print_options=['summary']))
    #     qa_with_gen_cyphers.save_to_disk(gen_save_dir)

if __name__ == '__main__':
    main()
