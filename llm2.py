import time
import argparse
import torch
from datasets import load_from_disk
from unsloth import FastLanguageModel, is_bfloat16_supported
from unsloth.chat_templates import train_on_responses_only
from trl import SFTConfig, SFTTrainer, DataCollatorForCompletionOnlyLM

from sequence_ranker import SequenceRanker
from compute_metrics import compute_metrics

# def format_prompt(data, tokenizer, max_seq_length, train=False):
#     instruction = (
#         "Given the information below, return the correct nodes for the following question: {question}\n"
#         "Retrieved information:\n{info}\n"
#         "Answer: "
#     )
#     response = "{answers}"
#     chat = [{"role": "user", "content": instruction.format(question=data['question'], info=data['info'])}]
#     if train:
#         chat.append({"role": "model", "content": response.format(answers=data['answer'])})
#
#     prompt = '\n'.join([f"<|start_header_id|>{item['role']}<|end_header_id|>\n{item['content']}" for item in chat])
#     #prompt = tokenizer.apply_chat_template(chat, tokenize=False, padding=True, max_length=max_seq_length)
#     data['text'] = prompt #do check if still adds <bos> here and at tokenizer
#     data['num_tokens'] = len(tokenizer(prompt))
#     return data

def formatting_prompts_func(example):
    output_texts = []
    instruction = ("Given the information below, return the correct nodes for the following question: {question}\n"
                   "Retrieved information:\n{info}\n")
    for question, info, answer in zip(example['question'], example['info'], example['answer']):
        text = (f"<|start_header_id|>user<|end_header_id|>\n{instruction.format(question=question, info=info)}\n"
                f"<|start_header_id|>model<|end_header_id|>\n{answer}")
        output_texts.append(text)
    return output_texts

def train(model, tokenizer, train_dataset, valid_dataset, max_seq_length, model_save_dir, do_train_on_responses_only=False):
    response_template = "<|start_header_id|>model<|end_header_id|>\n"
    collator = DataCollatorForCompletionOnlyLM(response_template, tokenizer=tokenizer)
    sft_config = SFTConfig(
        auto_find_batch_size=True,
        dataset_num_proc=8,
        bf16=True,
        num_train_epochs=3,  # Set this for 1 full training run.
        gradient_accumulation_steps=4,
        warmup_steps=5,
        learning_rate=2e-5,#2e-4, #what is it for g-retriever?
        optim="adamw_8bit",
        weight_decay = 0.01,
        lr_scheduler_type = "linear",
        logging_steps=1,
        #seed=3407,
        #report_to="none",  # Use this for WandB etc
        eval_strategy="epoch",
        save_strategy="epoch",
        # eval_strategy='steps',
        # eval_steps=20,
        #save_strategy='steps',
        #save_steps=500,
        output_dir=model_save_dir,
        load_best_model_at_end=True,
    )

    trainer = SFTTrainer(
        model=model,
        args=sft_config,
        tokenizer=tokenizer,
        train_dataset= train_dataset,
        eval_dataset = valid_dataset,
        dataset_text_field="text",
        max_seq_length=max_seq_length,
        dataset_num_proc=8,
        packing=False,  # Can make training 5x faster for short sequences.
        formatting_func=formatting_prompts_func,
        data_collator=collator,
    )

    # Train
    if do_train_on_responses_only:
        trainer = train_on_responses_only(
            trainer,
            instruction_part="<|start_header_id|>user<|end_header_id|>\n", #Must double check that this is correct!
            response_part="<|start_header_id|>model<|end_header_id|>\n",
        )
    trainer.train()

def add_ranked_sequences(data: dict, sequence_ranker, beam_width):
    prompt = ...
    possible_sequences = ...
    data['ranks'] = sequence_ranker.rank_sequences(prompt, possible_sequences, beam_width=beam_width)
    return data

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train', action='store_true') #Should be false by default, true if given
    parser.add_argument('--evaluate', action='store_true')
    parser.add_argument('--generate', action='store_true')
    parser.add_argument('--model_dir', type=str, default="llama8bi")  # use -Instruct?
    parser.add_argument('--model_save_dir', type=str, default=None)
    parser.add_argument('--train_data_dir', type=str, default='prime-data/qa_with_train_prompts')
    parser.add_argument('--eval_data_dir', type=str, default='prime-data/qa_with_valid_prompts')
    parser.add_argument('--eval_save_dir', type=str, default=None)
    parser.add_argument('--pred_save_dir', type=str, default=None)
    args = parser.parse_args()

    do_train = args.train
    do_evaluate = args.evaluate
    do_generate = args.generate
    model_save_dir = args.model_save_dir
    train_data_dir = args.train_data_dir
    eval_data_dir = args.eval_data_dir
    eval_save_dir = args.eval_save_dir
    pred_save_dir = args.pred_save_dir

    if args.model_dir == 'llama8bi':
        model_dir = "meta-llama/Llama-3.1-8B-Instruct"
    else:
        raise Exception("Unknown model")

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_dir,
        max_seq_length=None, # max_seq_length,
        dtype=torch.bfloat16,
        load_in_4bit=True,
    )
    tokenizer.padding = True
    tokenizer.pad_token = '<|finetune_right_pad_id|>'

    model = FastLanguageModel.get_peft_model(
        model=model,  # adapter name??
        r=16,  # Choose any number > 0 ! Suggested 8, 16, 32, 64, 128
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj", ],
        lora_alpha=16,
        lora_dropout=0,  # Supports any, but = 0 is optimized
        bias="none",  # Supports any, but = "none" is optimized
        # [NEW] "unsloth" uses 30% less VRAM, fits 2x larger batch sizes!
        use_gradient_checkpointing="unsloth",  # True or "unsloth" for very long context
        random_state=3407,
        use_rslora=False,  # We support rank stabilized LoRA
        loftq_config=None,  # And LoftQ
    )

    if do_train:
        max_seq_length = 11_000
        n_train = 1_000; n_eval = 200
        qa_with_train_prompts = load_from_disk(train_data_dir)\
            .filter(lambda _, i: i < n_train, with_indices=True)
        qa_with_eval_prompts = load_from_disk(eval_data_dir) \
            .filter(lambda _, i: i < n_eval, with_indices=True)
        # Put in train mode
        model_save_dir = f"./output-{time.time()}" if model_save_dir is None else model_save_dir

        train(model=model, tokenizer=tokenizer, train_dataset=qa_with_train_prompts, valid_dataset=qa_with_eval_prompts,
              max_seq_length=max_seq_length ,model_save_dir=model_save_dir, do_train_on_responses_only=False)

    #Put into inference mode? No grad, padding side: left?
    if do_evaluate:
        qa_with_eval_prompts = load_from_disk(eval_data_dir)\
            .filter(lambda _, i: i < 100, with_indices=True)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        start_of_generation_tokens = "<|start_header_id|>model<|end_header_id|>\n"
        end_of_generation_token = tokenizer.eos_token
        sequence_ranker = SequenceRanker(model=model, tokenizer=tokenizer, device=device,
                                         start_of_generation_tokens=start_of_generation_tokens,
                                         end_of_generation_token=end_of_generation_token)
        beam_width = 1_000  # big number
        qa_with_eval_prompts.map(lambda x: add_ranked_sequences(x, sequence_ranker, beam_width), num_proc=1)
        metrics = ['recall','mrr','hits@5',]

        compute_metrics(preds=qa_with_eval_prompts['ranks'], labels=qa_with_eval_prompts['answers'], metrics=metrics)

    if do_generate:
        pass

if __name__ == '__main__':
    main()