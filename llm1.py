import time
import copy
import argparse
import numpy as np
import torch
from transformers import StaticCache, BitsAndBytesConfig, GenerationConfig, DynamicCache, LlamaForCausalLM, \
    AutoModelForCausalLM, PreTrainedTokenizerBase, AutoTokenizer
from datasets import load_from_disk
from peft import LoraConfig, get_peft_model
from trl import SFTConfig, SFTTrainer

from logits_processor import MyLogitsProcessorList

#START_OF_GENERATION_TOKENS = "<|start_header_id|>assistant<|end_header_id|>"
#END_OF_GENERATION_TOKEN = "<|end_of_text|>"
#PAD_TOKEN = ... #? what's best?
START_OF_GENERATION_TOKENS = "<start_of_turn>assistant\n"
END_OF_GENERATION_TOKEN = "<eos>"#"<end_of_turn>"



def sort_cyphers(data: dict) -> dict:
    cyphers, hits, num_results = data['cyphers'], data['hits'], data['num_results']
    data['cyphers'], data['hits'], data['num_results'] = zip(*sorted(zip(cyphers, hits, num_results), key=lambda x: (-x[1],x[2])))
    return data


def best_label_is_good(data: dict, lowest_recall = 1, lowest_precision = .1) -> bool:
    sorted_data = sort_cyphers(data)
    recall = sorted_data['hits'][0]/sorted_data['num_results'][0]
    precision = sorted_data['hits'][0]/len(eval(sorted_data['answer_ids']))
    return recall >= lowest_recall and precision >= lowest_precision

def formatting_func(data: dict, instruction = None, add_label=True) -> str: #Not used yet
    prompt = ""#"<|begin_of_text|>"
    # if instruction is not None:
    #     prompt += f"<|start_header_id|>system<|end_header_id|>{instruction}<|eot_id|>"
    # prompt += f"<|start_header_id|>user<|end_header_id|>{data['question']}<|eot_id|>" + START_OF_GENERATION_TOKENS
    # if add_label:
    #     answer = sort_cyphers(data)['cyphers'][0]
    #     return prompt + answer + END_OF_GENERATION_TOKEN
    # else:
    #     return prompt
    prompt = f"<start_of_turn>user\n{data['question']}<end_of_turn>" + START_OF_GENERATION_TOKENS
    if add_label:
        answer = sort_cyphers(data)['cyphers'][0]
        prompt += answer+END_OF_GENERATION_TOKEN
    return prompt



def predict_cyphers(data: dict, model: LlamaForCausalLM, tokenizer: PreTrainedTokenizerBase, device, beam_width: int,
                    base_prompt: str = "", base_prompt_cache: DynamicCache = None) -> list[str]:
    possible_queries = data['cyphers']
    beam_width = min(beam_width, len(possible_queries))
    max_new_tokens = max([len(tokenizer.tokenize(query))+10 for query in possible_queries])
    generation_config = GenerationConfig(num_return_sequences=beam_width, num_beams=beam_width,
                                         max_new_tokens=max_new_tokens, early_stopping=True, do_sample=False,
                                         remove_invalid_values=True)

    allowed_queries_ids = [
        tokenizer(query + END_OF_GENERATION_TOKEN, add_special_tokens=False, return_tensors="pt").input_ids.to(device)
        for query in possible_queries]
    starting_ids = tokenizer(START_OF_GENERATION_TOKENS, add_special_tokens=False, return_tensors="pt").input_ids[0].to(device)
    logits_processor_list = MyLogitsProcessorList(allowed_queries_ids=allowed_queries_ids, starting_ids=starting_ids)

    # To not get dimension error. Does not work with StaticCache which can be faster
    #new_cache = tuple(tuple(torch.vstack(beam_width * [x]) for x in xs) for xs in base_prompt_cache) if base_prompt_cache is not None else None
    #base_prompt_cache = new_cache

    prompt = formatting_func(data=data, add_label=False)
    tokenized_question = tokenizer(prompt ,return_tensors="pt").to(device)
    outputs = model.generate(**tokenized_question,
                             generation_config=generation_config,
                             logits_processor=logits_processor_list,
                             #past_key_values=copy.deepcopy(base_prompt_cache),
                             #use_cache=True
                             )
    decoded_outputs = tokenizer.batch_decode(outputs, skip_special_tokens=False)
    top_cypher_queries = [text.split(START_OF_GENERATION_TOKENS)[-1].split(END_OF_GENERATION_TOKEN)[0] for text in decoded_outputs]
    return top_cypher_queries


def add_predicted_cypher(data: dict, device, model: LlamaForCausalLM, tokenizer: PreTrainedTokenizerBase,
                         beam_width: int, base_prompt: str, base_prompt_cache: DynamicCache = None, print_options = []) -> dict:
    data['top_cyphers'] = predict_cyphers(data=data, model=model, tokenizer=tokenizer, device=device,
                                          beam_width=beam_width, base_prompt=base_prompt,
                                          base_prompt_cache=base_prompt_cache)
    if 'summary' in print_options:
        print(f"Top cyphers: {data['top_cyphers']}\n"
              f"Total #cyphers:     {len(data['cyphers'])}\n"
              f"Generated #cyphers: {len(data['top_cyphers'])}\n"
              f"Gen valid #cyphers: {len(set(data['top_cyphers']).intersection(data['cyphers']))}\n")

    if 'add_details' in print_options:
        sorted_data = sort_cyphers(data)
        max_recall = 0
        print(f"Question: {data['question']}")
        for top_cypher in data['top_cyphers']:
            try:
                i = sorted_data['cyphers'].index(top_cypher)
                precision = sorted_data['hits'][i] / sorted_data['num_results'][i]
                recall = sorted_data['hits'][i] / len(eval(data['answer_ids']))
                if 'predicted_recall_at_1' not in data.keys():
                    data['rank@1'] = i+1
                    data['num_nodes_at_1'] = sorted_data['num_results'][i]
                    data['predicted_recall_at_1'] = recall
                max_recall = max(max_recall, recall)
                print(f"Rank: {i+1}   Precision: {precision:.3f}   Recall: {recall:.3f}   Cypher query: {top_cypher}")
            except ValueError:
                print("not expected...")

        print(f"Max recall: {max_recall}, Recall@1: {data.get('predicted_recall_at_1','-')}, "
              f"#nodes@1: {data.get('num_nodes_at_1','-')} Rank@1: {data.get('rank@1', '-')}\n")
        data['predicted_max_recall'] = max_recall

    if 'gpu_info' in print_options:
        print("torch.cuda.memory_allocated: %fGB" % (torch.cuda.memory_allocated(0) / 1024 / 1024 / 1024))
        print("torch.cuda.memory_reserved: %fGB" % (torch.cuda.memory_reserved(0) / 1024 / 1024 / 1024))
        print("torch.cuda.max_memory_reserved: %fGB" % (torch.cuda.max_memory_reserved(0) / 1024 / 1024 / 1024))

    return data


def train(model: AutoModelForCausalLM, tokenizer: PreTrainedTokenizerBase, data_path: str, model_save_dir: str, base_prompt: str):
    # Load data
    qa_with_train_cyphers = load_from_disk(data_path)
    qa_with_train_cyphers.pop('test')
    qa_with_supervised_prompts = qa_with_train_cyphers\
        .filter(lambda x: best_label_is_good(x, lowest_recall=1, lowest_precision=.1))\
        .map(lambda x: x | {'text' : formatting_func(x, add_label=True)})
        #.map(lambda x: format_prompt(x, base_prompt=base_prompt, train=True))
    max_seq_len = max([len(tokenizer.encode(x['text'])) for x in qa_with_supervised_prompts['train']]) + 10

    # lora_config = LoraConfig(r=64, lora_alpha=64, target_modules=None, lora_dropout=0.05, bias="none",
    #                          task_type="CAUSAL_LM", )
    lora_config = LoraConfig(r=16, lora_alpha=16, target_modules=None, lora_dropout=0.05, bias="none",
                             task_type="CAUSAL_LM", )
    peft_model = get_peft_model(model=model, peft_config=lora_config)

    sft_config = SFTConfig(auto_find_batch_size=True,
                           gradient_accumulation_steps=1, #8
                           dataset_num_proc=8,
                           num_train_epochs=3,
                           learning_rate=2e-5,
                           optim="paged_adamw_8bit",
                           max_seq_length=max_seq_len,
                           evaluation_strategy="epoch",
                           save_steps=5,
                           save_total_limit=1,
                           save_strategy="epoch",
                           logging_steps=5,
                           output_dir=model_save_dir,
                           load_best_model_at_end=True,
                           )

    trainer = SFTTrainer(model=peft_model,
                         args=sft_config,
                         processing_class=tokenizer,
                         train_dataset=qa_with_supervised_prompts['train'],
                         eval_dataset=qa_with_supervised_prompts['valid'],
                         # formatting_func=None #format here instead of mapping all in beforehand
                         )
    trainer.train()  # Train on response only is worse?


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train', action='store_true') #Should be false by default, true if given
    parser.add_argument('--evaluate', action='store_true')
    parser.add_argument('--generate', action='store_true')
    parser.add_argument('--model_dir', type=str, default="meta-llama/Llama-3.1-8B-Instruct")  # use -Instruct?
    parser.add_argument('--model_save_dir', type=str, default=None)
    parser.add_argument('--data_dir', type=str, default='prime-data/qa_with_train_cyphers')
    parser.add_argument('--eval_save_dir', type=str, default=None)
    parser.add_argument('--pred_save_dir', type=str, default=None)
    parser.add_argument('--use_base_prompt', action='store_true')
    parser.add_argument('--beam_width', type=int, default=5)
    parser.add_argument('--eval_fraction', type=float, default=1)
    args = parser.parse_args()

    do_train = args.train
    do_evaluate = args.evaluate
    do_generate = args.generate
    model_dir = args.model_dir
    model_save_dir = args.model_save_dir
    data_dir = args.data_dir
    eval_save_dir = args.eval_save_dir
    pred_save_dir = args.pred_save_dir
    use_base_prompt = args.use_base_prompt
    beam_width = args.beam_width
    eval_fraction = args.eval_fraction

    if args.model_dir == 'llama8b':
        model_dir = "meta-llama/Llama-3.1-8B"
    #elif args.model_dir == 'llamaT2C':
        #model_dir = "tomasonjo/text2cypher-demo-16bit"
        #model_dir = "neo4j/text2cypher-llama3_1_8B_instruct-unsloth-finetuned-2024v1"
        #model_dir = "neo4j/neo4j_llama318b_finetuned_merged_oct24"
    elif args.model_dir == 'llama1bi':
        model_dir = "meta-llama/Llama-3.2-1B-Instruct"
    elif args.model_dir == 'llama8bi':
        model_dir = "meta-llama/Llama-3.1-8B-Instruct"
    elif args.model_dir == 'gemmaT2C':
        model_dir = "neo4j/text2cypher-gemma-2-9b-it-finetuned-2024v1"
    # elif args.model_dir == 'old_gemma':
    #     model_dir = "old/neo4j/text2cypher-gemma-2-9b-it-finetuned-2024v1-unsloth-ft-246"
    # elif args.model_dir == 'epfl':
        #model_dir = "epfl-llm/meditron-7b"
        # model_dir = "llama8-feb-18_3/checkpoint-546"
    else:
        raise Exception("Unknown model")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Model
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
    )
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    #tokenizer.pad_token_id = tokenizer.eos_token_id # don't use eos token as padding??
    #tokenizer.pad_token = '<|finetune_right_pad_id|>'
    #model = LlamaForCausalLM.from_pretrained(
    tokenizer.padding_side = 'right'
    model = AutoModelForCausalLM.from_pretrained(
        model_dir,
        quantization_config=bnb_config if device != torch.device('cpu') else None,
        torch_dtype=torch.bfloat16,
        attn_implementation="eager",
        low_cpu_mem_usage=True,
    ).to(device)
    if args.model_dir == 'gemmaT2C':
        model.padding_side = 'right'

    # Load data
    qa_with_train_cyphers = load_from_disk('prime-data/qa_with_train_cyphers')

    if use_base_prompt:
        # Create base prompt
        schema = "Relationships without direction (:Disease)-[:OFF_LABEL_USE]-(:Drug), (:Drug)-[:OFF_LABEL_USE]-(:Disease), (:Disease)-[:ASSOCIATED_WITH]-(:GeneOrProtein), (:GeneOrProtein)-[:ASSOCIATED_WITH]-(:Disease), (:Disease)-[:PARENT_CHILD]-(:Disease), (:Disease)-[:PHENOTYPE_ABSENT]-(:EffectOrPhenotype), (:EffectOrPhenotype)-[:PHENOTYPE_ABSENT]-(:Disease), (:Disease)-[:PHENOTYPE_PRESENT]-(:EffectOrPhenotype), (:EffectOrPhenotype)-[:PHENOTYPE_PRESENT]-(:Disease), (:Disease)-[:LINKED_TO]-(:Exposure), (:Exposure)-[:LINKED_TO]-(:Disease), (:Disease)-[:CONTRAINDICATION]-(:Drug), (:Drug)-[:CONTRAINDICATION]-(:Disease), (:Disease)-[:INDICATION]-(:Drug), (:Drug)-[:INDICATION]-(:Disease), (:GeneOrProtein)-[:PPI]-(:GeneOrProtein), (:GeneOrProtein)-[:INTERACTS_WITH]-(:MolecularFunction), (:MolecularFunction)-[:INTERACTS_WITH]-(:GeneOrProtein), (:GeneOrProtein)-[:INTERACTS_WITH]-(:BiologicalProcess), (:BiologicalProcess)-[:INTERACTS_WITH]-(:GeneOrProtein), (:GeneOrProtein)-[:INTERACTS_WITH]-(:CellularComponent), (:CellularComponent)-[:INTERACTS_WITH]-(:GeneOrProtein), (:GeneOrProtein)-[:EXPRESSION_PRESENT]-(:Anatomy), (:Anatomy)-[:EXPRESSION_PRESENT]-(:GeneOrProtein), (:GeneOrProtein)-[:EXPRESSION_ABSENT]-(:Anatomy), (:Anatomy)-[:EXPRESSION_ABSENT]-(:GeneOrProtein), (:GeneOrProtein)-[:INTERACTS_WITH]-(:Pathway), (:Pathway)-[:INTERACTS_WITH]-(:GeneOrProtein), (:GeneOrProtein)-[:TARGET]-(:Drug), (:Drug)-[:TARGET]-(:GeneOrProtein), (:GeneOrProtein)-[:TRANSPORTER]-(:Drug), (:Drug)-[:TRANSPORTER]-(:GeneOrProtein), (:GeneOrProtein)-[:CARRIER]-(:Drug), (:Drug)-[:CARRIER]-(:GeneOrProtein), (:GeneOrProtein)-[:ENZYME]-(:Drug), (:Drug)-[:ENZYME]-(:GeneOrProtein), (:GeneOrProtein)-[:ASSOCIATED_WITH]-(:EffectOrPhenotype), (:EffectOrPhenotype)-[:ASSOCIATED_WITH]-(:GeneOrProtein), (:GeneOrProtein)-[:INTERACTS_WITH]-(:Exposure), (:Exposure)-[:INTERACTS_WITH]-(:GeneOrProtein), (:MolecularFunction)-[:PARENT_CHILD]-(:MolecularFunction), (:MolecularFunction)-[:INTERACTS_WITH]-(:Exposure), (:Exposure)-[:INTERACTS_WITH]-(:MolecularFunction), (:Drug)-[:SYNERGISTIC_INTERACTION]-(:Drug), (:Drug)-[:SIDE_EFFECT]-(:EffectOrPhenotype), (:EffectOrPhenotype)-[:SIDE_EFFECT]-(:Drug), (:Pathway)-[:PARENT_CHILD]-(:Pathway), (:Anatomy)-[:PARENT_CHILD]-(:Anatomy), (:EffectOrPhenotype)-[:PARENT_CHILD]-(:EffectOrPhenotype), (:BiologicalProcess)-[:PARENT_CHILD]-(:BiologicalProcess), (:BiologicalProcess)-[:INTERACTS_WITH]-(:Exposure), (:Exposure)-[:INTERACTS_WITH]-(:BiologicalProcess), (:CellularComponent)-[:PARENT_CHILD]-(:CellularComponent), (:CellularComponent)-[:INTERACTS_WITH]-(:Exposure), (:Exposure)-[:INTERACTS_WITH]-(:CellularComponent), (:Exposure)-[:PARENT_CHILD]-(:Exposure)"
        instruction = "Generate Cypher statement to query a graph database. Use only the provided relationship types and properties in the schema."
        base_prompt = f"<bos><start_of_turn>user\n{instruction}\nSchema: {schema}"
        # Pre-process base prompt
        inputs = tokenizer(base_prompt, return_tensors="pt")
        # prompt_cache = StaticCache(config=model.config, max_batch_size=1, max_cache_len=1024, device=device,
        #                            dtype=torch.bfloat16)
        with torch.no_grad():
            base_prompt_cache = model(**inputs, cache_implementation="dynamic").past_key_values
            #base_prompt_cache = model(**inputs, past_key_values=prompt_cache).past_key_values
    else:
        base_prompt = ""
        base_prompt_cache = None

    if do_train:
        # Put in train mode
        model_save_dir = f"./output-{time.time()}" if model_save_dir is None else model_save_dir
        train(model=model, tokenizer=tokenizer, data_path=data_dir, model_save_dir=model_save_dir, base_prompt=base_prompt)

    #Put into inference mode? No grad, padding side: left?
    if do_evaluate:
        qa_with_train_cyphers = load_from_disk(data_dir)['valid']
        qa_with_train_cyphers = qa_with_train_cyphers.filter(lambda _,i: i < int(len(qa_with_train_cyphers)*eval_fraction), with_indices=True)
        qa_with_evaluation_result = qa_with_train_cyphers.map(lambda data: add_predicted_cypher(data, device, model, tokenizer, beam_width, base_prompt, base_prompt_cache, print_options=['add_details', 'gpu_info']))
        print(f"Avg recall@1: {np.mean(qa_with_evaluation_result['predicted_recall_at_1']):.2f}    Avg top recall of 5: {np.mean(qa_with_evaluation_result['predicted_max_recall'])}    Avg #nodes@1 {np.mean(qa_with_evaluation_result['num_nodes_at_1'])}")
        eval_save_dir = "prime-data/qa_with_eval_cyphers" if eval_save_dir is None else eval_save_dir
        qa_with_evaluation_result.save_to_disk(eval_save_dir)

    if do_generate:
        # Generate cypher queries for all questions
        # Only for test set today
        qa_with_train_cyphers = qa_with_train_cyphers['test']
        qa_with_gen_cyphers = qa_with_train_cyphers.map(
            lambda data: add_predicted_cypher(data, device=device, model=model, tokenizer=tokenizer,
                                              beam_width=beam_width, base_prompt=base_prompt,
                                              base_prompt_cache=base_prompt_cache, print_options=['summary']))
        pred_save_dir = "prime-data/qa_with_gen_cyphers" if pred_save_dir is None else pred_save_dir
        qa_with_gen_cyphers.save_to_disk(pred_save_dir)

if __name__ == '__main__':
    main()
