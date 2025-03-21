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


def sort_cyphers(data: dict) -> dict:
    cyphers, hits, num_results = data['cypher_queries'], data['hits'], data['num_results']
    data['cypher_queries'], data['hits'], data['num_results'] = zip(
        *sorted(zip(cyphers, hits, num_results), key=lambda x: (-x[1], x[2])))
    return data


def best_label_is_good(data: dict, lowest_recall=1, lowest_precision=.1) -> bool:
    sorted_data = sort_cyphers(data)
    precision = sorted_data['hits'][0] / sorted_data['num_results'][0]
    recall = sorted_data['hits'][0] / len(sorted_data['answer_ids'])
    return recall >= lowest_recall and precision >= lowest_precision


def formatting_func(data: dict, add_label=True) -> str:  # Not used yet
    prompt = f"<start_of_turn>user\n{data['question']}<end_of_turn>" + START_OF_GENERATION_TOKENS
    if add_label:
        answer = sort_cyphers(data)['cypher_queries'][0]
        prompt += answer + END_OF_GENERATION_TOKEN
    return prompt


def add_predicted_cypher(data: dict, beam_width: int, sequence_ranker: SequenceRanker, print_options=[],) -> dict:
    prompt = formatting_func(data=data, add_label=False)
    possible_sequences = data['cypher_queries']
    data['top_cypher_queries'] = sequence_ranker.rank_sequences(prompt=prompt, possible_sequences=possible_sequences, max_beam_width=beam_width)

    if 'summary' in print_options:
        print(f"Top cyphers: {data['top_cypher_queries']}\n"
              f"Total #cyphers:     {len(data['cypher_queries'])}\n"
              f"Generated #cyphers: {len(data['top_cypher_queries'])}\n"
              f"Gen valid #cyphers: {len(set(data['top_cypher_queries']).intersection(data['cypher_queries']))}\n")

    if 'add_details' in print_options:
        sorted_data = sort_cyphers(data)
        max_recall = 0
        print(f"Question: {data['question']}")
        for top_cypher in data['top_cypher_queries']:
            try:
                i = sorted_data['cypher_queries'].index(top_cypher)
                precision = sorted_data['hits'][i] / sorted_data['num_results'][i]
                recall = sorted_data['hits'][i] / len(data['answer_ids'])
                if 'predicted_recall_at_1' not in data.keys():
                    data['rank@1'] = i + 1
                    data['num_nodes_at_1'] = sorted_data['num_results'][i]
                    data['predicted_recall_at_1'] = recall
                max_recall = max(max_recall, recall)
                print(f"Rank: {i + 1}   Precision: {precision:.3f}   Recall: {recall:.3f}   Cypher query: {top_cypher}")
            except ValueError:
                print("not expected...")

        print(f"Max recall: {max_recall}, Recall@1: {data.get('predicted_recall_at_1', '-')}, "
              f"#nodes@1: {data.get('num_nodes_at_1', '-')} Rank@1: {data.get('rank@1', '-')}\n")
        data['predicted_max_recall'] = max_recall

    if 'gpu_info' in print_options:
        print("torch.cuda.memory_allocated: %fGB" % (torch.cuda.memory_allocated(0) / 1024 / 1024 / 1024))
        print("torch.cuda.memory_reserved: %fGB" % (torch.cuda.memory_reserved(0) / 1024 / 1024 / 1024))
        print("torch.cuda.max_memory_reserved: %fGB" % (torch.cuda.max_memory_reserved(0) / 1024 / 1024 / 1024))

    return data


def train(model: AutoModelForCausalLM, tokenizer: PreTrainedTokenizerBase, train_dataset: Dataset,
          eval_dataset: Dataset, model_save_dir: str):
    # Load data
    max_seq_len = max([len(tokenizer.encode(x['text'])) for x in train_dataset]) + 10

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

    trainer = SFTTrainer(model=model,
                         args=sft_config,
                         processing_class=tokenizer,
                         train_dataset=train_dataset,
                         eval_dataset=eval_dataset,
                         # formatting_func=None #format here instead of mapping all in beforehand
                         )
    trainer.train()  # Train on response only is worse?

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train', action='store_true')
    parser.add_argument('--evaluate', action='store_true')
    parser.add_argument('--generate', action='store_true')
    parser.add_argument('--model_dir', type=str,
                        default="neo4j/text2cypher-gemma-2-9b-it-finetuned-2024v1")
    parser.add_argument('--dataset', type=str, default=None)
    parser.add_argument('--adapter_dir', type=str, default=None)
    parser.add_argument('--model_save_dir', type=str, default=None)
    parser.add_argument('--data_dir', type=str, default=None)
    parser.add_argument('--train_data_dir', type=str, default=None)
    parser.add_argument('--valid_data_dir', type=str, default=None)
    parser.add_argument('--gen_data_dir', type=str, default=None)
    parser.add_argument('--eval_save_dir', type=str, default=None)
    parser.add_argument('--gen_save_dir', type=str, default=None)
    # parser.add_argument('--eval_save_dir', type=str, default="prime-data/qa_with_eval_cyphers")
    # parser.add_argument('--gen_save_dir', type=str, default="prime-data/qa_with_gen_cyphers")
    parser.add_argument('--use_base_prompt', action='store_true')
    parser.add_argument('--beam_width', type=int, default=5)
    parser.add_argument('--eval_fraction', type=float, default=1)
    parser.add_argument('--gen_fraction', type=float, default=1)
    args = parser.parse_args()

    do_train = args.train
    do_evaluate = args.evaluate
    do_generate = args.generate

    dataset_name = args.dataset
    if dataset_name not in ['prime', 'mag']:
        raise ValueError(f"Dataset {dataset_name} not supported. Select 'prime' or 'mag'")

    model_dir = args.model_dir
    adapter_dir = args.adapter_dir
    if args.model_save_dir is None:
        model_save_dir = f"./llm1-{int(time.time())}"
    else:
        model_save_dir = args.model_save_dir

    if args.data_dir is None:
        if do_train and args.train_data_dir is None:
            raise ValueError("--data_dir or --train_data_dir must be specified if --train is True")
        if do_train and args.valid_data_dir is None:
            raise ValueError("--data_dir or --valid_data_dir must be specified if --train is True")
        if (do_evaluate or do_generate) and args.valid_save_dir is None:
            raise ValueError("--data_dir or --valid_save_dir must be specified if --evaluate is True")
    # if do_generate and args.gen_data_dir is None:
    #     raise ValueError("--gen_data_dir must be specified if --generate is True")
    train_data_dir = os.path.join(args.data_dir, 'train') if args.train_data_dir is None else args.train_data_dir
    valid_data_dir = os.path.join(args.data_dir, 'valid') if args.valid_data_dir is None else args.valid_data_dir
    # gen_data_dir = os.path.join(args.data_dir, 'valid') if args.gen_data_dir is None else args.gen_data_dir
    gen_data_dir = os.path.join(args.data_dir, 'test') if args.gen_data_dir is None else args.gen_data_dir

    eval_save_dir = args.eval_save_dir
    gen_save_dir = args.gen_save_dir if args.gen_save_dir is not None else f"{dataset_name}-data/qa_with_generated_cypher_queries_test"

    beam_width = args.beam_width
    eval_fraction = args.eval_fraction
    gen_fraction = args.gen_fraction

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Model
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
    )
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    tokenizer.padding_side = 'right'

    model = AutoModelForCausalLM.from_pretrained(
        model_dir,
        quantization_config=bnb_config if device != torch.device('cpu') else None,
        torch_dtype=torch.bfloat16,
        attn_implementation="eager",
        low_cpu_mem_usage=True,
    ).to(device)
    model.padding_side = 'right'

    if adapter_dir is None:
        lora_config = LoraConfig(r=64, lora_alpha=64, target_modules=None, lora_dropout=0.05, bias="none", task_type="CAUSAL_LM", )
        model = get_peft_model(model=model, peft_config=lora_config)
    else:
        model = PeftModel.from_pretrained(model, adapter_dir)

    if do_train:
        # Put in train mode
        qa_with_supervised_prompts_train = load_from_disk(train_data_dir)\
            .filter(lambda x: best_label_is_good(x, lowest_recall=1, lowest_precision=.1))\
            .map(lambda x: x | {'text': formatting_func(x, add_label=True)})
        qa_with_supervised_prompts_valid = load_from_disk(valid_data_dir) \
            .filter(lambda x: best_label_is_good(x, lowest_recall=1, lowest_precision=.1)) \
            .map(lambda x: x | {'text': formatting_func(x, add_label=True)})
        train(model=model, tokenizer=tokenizer, train_dataset=qa_with_supervised_prompts_train,
              eval_dataset=qa_with_supervised_prompts_valid, model_save_dir=model_save_dir)

    # Put into inference mode? No grad, padding side: left?
    sequence_ranker = SequenceRanker(model, tokenizer, device, START_OF_GENERATION_TOKENS, END_OF_GENERATION_TOKEN)

    if do_evaluate:
        qa_with_cyphers = load_from_disk(valid_data_dir)
        qa_with_evaluation_result = qa_with_cyphers\
            .filter(lambda _, i: i < int(len(qa_with_cyphers) * eval_fraction), with_indices=True) \
            .map(lambda data: add_predicted_cypher(data, sequence_ranker=sequence_ranker, beam_width=beam_width,
                                                   print_options=['add_details', 'gpu_info']))
        print(f"Avg recall@1: {np.mean(qa_with_evaluation_result['predicted_recall_at_1']):.2f}    "
              f"Avg top recall of 5: {np.mean(qa_with_evaluation_result['predicted_max_recall'])}    "
              f"Avg #nodes@1 {np.mean(qa_with_evaluation_result['num_nodes_at_1'])}")
        qa_with_evaluation_result.save_to_disk(eval_save_dir)

    if do_generate:
        # Generate cypher queries for all questions
        qa_with_cyphers = load_from_disk(gen_data_dir)
        qa_with_gen_cyphers = qa_with_cyphers \
            .filter(lambda _, i: i < int(len(qa_with_cyphers) * gen_fraction), with_indices=True) \
            .map(lambda data: add_predicted_cypher(data, sequence_ranker=sequence_ranker, beam_width=beam_width,))
                                                   #print_options=['summary']))
        qa_with_gen_cyphers.save_to_disk(gen_save_dir)


if __name__ == '__main__':
    main()
