import unsloth
from unsloth import FastLanguageModel
from unsloth.chat_templates import train_on_responses_only

import os
import time
import argparse
import torch
import peft
from datasets import load_from_disk
from transformers import GenerationConfig
from trl import SFTConfig, SFTTrainer, DataCollatorForCompletionOnlyLM

from compute_metrics import compute_metrics


INSTRUCTION_TEMPLATE = "<|start_header_id|>user<|end_header_id|>\n"
RESPONSE_TEMPLATE = "<|start_header_id|>model<|end_header_id|>\n"
EOS = "<|eot_id|>"
RIGHT_PAD_TOKEN = '<|finetune_right_pad_id|>'
ANSWER_SEPARATOR = '|' #replace with special token?
PRIME_MAX_SEQUENCE_LENGTH = 15_000
MAG_MAX_SEQUENCE_LENGTH = 15_000
MAX_NEW_TOKENS = 100
INSTRUCTION = ("Given the information below, return the correct nodes for the following question: {question}\n"
               "Retrieved information:\n{info}\n")


def formatting_prompts_func_train(example):
    output_texts = []
    instruction = ("Given the information below, return the correct nodes for the following question: {question}\n"
                   "Retrieved information:\n{info}\n")
    for question, info, answer_names in zip(example['question'], example['info'], example['answer_names']):
        answer = ANSWER_SEPARATOR.join(answer_names)
        text = (f"{INSTRUCTION_TEMPLATE}{instruction.format(question=question, info=info)}\n"
                f"{RESPONSE_TEMPLATE}{answer}"+EOS)
        output_texts.append(text)
    return output_texts


class LLM2:
    def __init__(self, model_dir, properties, max_sequence_length):
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=model_dir,
            max_seq_length=max_sequence_length,  # None
            dtype=torch.bfloat16,
            load_in_4bit=True,
        )
        tokenizer.padding = True
        tokenizer.pad_token = RIGHT_PAD_TOKEN
        tokenizer.padding_side = 'right'

        if type(model.base_model) is not peft.LoraModel: #Otherwise it already has LoRA params
            model = FastLanguageModel.get_peft_model(
                model=model,
                r=64,
                target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj", ],
                lora_alpha=64,
                bias="none",
            )

        self.tokenizer = tokenizer
        self.model = model
        self.properties = properties


    def format_data(self, nodes_data: list[dict]):
        info = '\n\n'.join([
            '\n'.join([f"{prop}: {node_data[prop]}" for prop in self.properties])
            for node_data in nodes_data])
        return info


    def format_prompt(self, question, nodes_data: list[dict]):
        retrieved_info = self.format_data(nodes_data)
        text = INSTRUCTION_TEMPLATE + INSTRUCTION.format(question=question, info=retrieved_info) + '\n' + RESPONSE_TEMPLATE
        return text


    def train(self, train_dataset, valid_dataset, max_seq_length, model_save_dir):
        collator = DataCollatorForCompletionOnlyLM(RESPONSE_TEMPLATE, tokenizer=self.tokenizer)
        sft_config = SFTConfig(
            per_device_train_batch_size=1,
            per_device_eval_batch_size=1,
            # auto_find_batch_size=True,
            dataset_num_proc=8,
            bf16=True,
            num_train_epochs=1,#3,  # Set this for 1 full training run.
            gradient_accumulation_steps=4,
            warmup_steps=5,
            learning_rate=2e-5,  # 2e-4, #what is it for g-retriever?
            optim="adamw_8bit",
            weight_decay=0.01,
            lr_scheduler_type="linear",
            logging_steps=1,
            eval_strategy="epoch",
            save_strategy="epoch",
            output_dir=model_save_dir,
            load_best_model_at_end=True,
        )

        trainer = SFTTrainer(
            model=self.model,
            args=sft_config,
            tokenizer=self.tokenizer,
            train_dataset=train_dataset,
            eval_dataset=valid_dataset,
            dataset_text_field="text",
            max_seq_length=max_seq_length,
            dataset_num_proc=8,
            packing=False,  # Can make training 5x faster for short sequences.
            formatting_func=formatting_prompts_func_train,
            data_collator=collator,
        )

        # Train
        trainer = train_on_responses_only(
            trainer,
            instruction_part=INSTRUCTION_TEMPLATE,
            response_part=RESPONSE_TEMPLATE,
        )
        trainer.train()


    def put_in_training_mode(self):
        raise NotImplemented


    def put_in_inference_mode(self):
        self.model = FastLanguageModel.for_inference(self.model)
        self.tokenizer.pad_token = self.tokenizer.eos_token


    def evaluate(self, eval_dataset_dir: str, eval_rate: float, metrics, eval_save_dir = None, add_more_answers=False):
        eval_dataset = load_from_disk(eval_dataset_dir)
        n_eval = int(eval_rate * len(eval_dataset))
        qa_with_answers = eval_dataset \
            .map(lambda x: x | {'data': str(x['data'])}) \
            .filter(lambda x: len(eval(x['data'])) > 0 and len(x['answer_names']) > 0) \
            .filter(lambda _, i: i < n_eval, with_indices=True) \
            .map(lambda x: x | {'predicted_answers': self.generate_answer(x['question'], eval(x['data']), add_more_answers=add_more_answers)},
                 num_proc=1)

        if eval_save_dir is not None:
            qa_with_answers.save_to_disk(eval_save_dir)

        compute_metrics(predss=qa_with_answers['predicted_answers'], labelss=qa_with_answers['answer_names'], metrics=metrics)


    def generate_answer(self, question:str, nodes_data: list[dict], add_more_answers=False):
        device = self.model.device

        generation_config = GenerationConfig(early_stopping=True, do_sample=False, max_new_tokens=MAX_NEW_TOKENS)

        prompt = self.format_prompt(question=question, nodes_data=nodes_data)
        tokenized_prompt = self.tokenizer(prompt, return_tensors="pt", add_special_tokens=False, return_token_type_ids=False).to(device)
        output = self.model.generate(**tokenized_prompt, generation_config=generation_config)[0]
        decoded_output = self.tokenizer.decode(output, skip_special_tokens=False)
        predicted_answers = decoded_output.split(RESPONSE_TEMPLATE)[-1].split('<|eot_id|>')[0].split(ANSWER_SEPARATOR)
        if add_more_answers:
            predicted_answers.extend([node_data['name'] for node_data in nodes_data if node_data['name'] not in predicted_answers])
        print(f"Generated: {decoded_output.split(RESPONSE_TEMPLATE)[-1]}")
        print(f"Parsed:    {ANSWER_SEPARATOR.join(predicted_answers)}")
        return predicted_answers

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default=None)

    parser.add_argument('--train', action='store_true') #Should be false by default, true if given
    parser.add_argument('--eval', action='store_true')
    parser.add_argument('--test', action='store_true')
    parser.add_argument('--add_more_answers', action='store_true')

    parser.add_argument('--train_rate', type=float, default=1)
    parser.add_argument('--eval_rate', type=float, default=1)
    parser.add_argument('--test_rate', type=float, default=1)

    parser.add_argument('--model_dir', type=str, default="meta-llama/Llama-3.1-8B-Instruct")  # use -Instruct?
    parser.add_argument('--adapter_dir', type=str, default=None)
    parser.add_argument('--model_save_name', type=str, default=None)

    args = parser.parse_args()

    train_data_dir = f"{args.dataset}-data/qa_with_retrieved_data/train"
    valid_data_dir = f"{args.dataset}-data/qa_with_retrieved_data/valid"
    test_data_dir = f"{args.dataset}-data/qa_with_retrieved_data/test"

    model_save_name = args.model_save_name if args.model_save_name is not None else f"llm2-{int(time.time()) % 100_000}"
    model_save_dir = os.path.join(f"{args.dataset}-models", model_save_name)

    match args.dataset:
        case 'prime':
            max_sequence_length = PRIME_MAX_SEQUENCE_LENGTH #calculate instead
            properties = ['pattern', 'details']
        case 'mag':
            max_sequence_length = MAG_MAX_SEQUENCE_LENGTH
            properties = ['pattern', 'name', 'abstract']
        case _:
            raise Exception(f"Unknown dataset: {args.dataset}")

    model_dir = args.model_dir if args.adapter_dir is None else args.adapter_dir
    llm2 = LLM2(model_dir, properties, max_sequence_length)

    if args.train:
        qa_with_retrieved_data_train = load_from_disk(train_data_dir)
        qa_with_retrieved_data_eval = load_from_disk(valid_data_dir)
        n_train = int(args.train_rate * len(qa_with_retrieved_data_train))
        n_eval = int(args.train_rate * len(qa_with_retrieved_data_eval))
        qa_with_retrieved_data_train = qa_with_retrieved_data_train \
            .filter(lambda x: len(eval(x['data'])) > 0 and len(x['answer_names']) > 0) \
            .filter(lambda _, i: i < n_train, with_indices=True) \
            .remove_columns(['q_emb', 'predicted_entities', 'cypher_queries', 'hits', 'num_results', 'top_cypher_queries', 'answer_ids', 'id']) \
            .map(lambda x: x | {'info': llm2.format_data(eval(x['data']))})#.map(lambda x: x | {'text' : llm2.formatting_prompts_func(x, add_label=True)})
        qa_with_retrieved_data_eval = qa_with_retrieved_data_eval \
            .filter(lambda x: len(eval(x['data'])) > 0 and len(x['answer_names']) > 0) \
            .filter(lambda _, i: i < n_eval, with_indices=True) \
            .remove_columns(['q_emb', 'predicted_entities', 'cypher_queries', 'hits', 'num_results', 'top_cypher_queries', 'answer_ids', 'id']) \
            .map(lambda x: x | {'info': llm2.format_data(eval(x['data']))})#.map(lambda x: x | {'text' : llm2.formatting_prompts_func(x, add_label=True)})

        llm2.train(train_dataset=qa_with_retrieved_data_train, valid_dataset=qa_with_retrieved_data_eval,
              model_save_dir=model_save_dir, max_seq_length=max_sequence_length)

    llm2.put_in_inference_mode()

    if args.eval:
        llm2.evaluate(eval_dataset_dir=valid_data_dir, eval_rate=args.eval_rate,
                      metrics=['f1', 'precision', 'recall', 'hit@1', 'hit@5', 'recall@20', 'mrr', 'num_nodes'])
    if args.test:
        llm2.evaluate(eval_dataset_dir=test_data_dir, eval_rate=args.test_rate,
                      metrics=['f1', 'precision', 'recall', 'hit@1', 'hit@5', 'recall@20', 'mrr', 'num_nodes'])

if __name__ == '__main__':
    main()
