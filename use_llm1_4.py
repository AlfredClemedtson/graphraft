import torch
from transformers import AutoTokenizer, StaticCache, CacheConfig, BitsAndBytesConfig, \
    GenerationConfig, LlamaForCausalLM, LogitsProcessor, AutoModelForCausalLM, DynamicCache, QuantizedCache
from datasets import load_from_disk
from tqdm import tqdm

BEAM_WIDTH = 10
MAX_LENGTH = 200 #Max num tokens in generated output (cypher query)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#device = 'cpu'

print(device)

# Model
#model_name = "neo4j/text2cypher-gemma-2-9b-it-finetuned-2024v1"
#model_name = "meta-llama/Llama-3.2-1B-Instruct"
model_name = "meta-llama/Llama-3.1-8B-Instruct"

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
)
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token_id = tokenizer.eos_token_id
model = AutoModelForCausalLM.from_pretrained(
#model = LlamaForCausalLM.from_pretrained(
    model_name,
    quantization_config=bnb_config,
    torch_dtype=torch.bfloat16,
    attn_implementation="flash_attention_2",
    low_cpu_mem_usage=True,
    #device_map="auto",??
).to(device)


class MyLogitsProcessor(LogitsProcessor):
    def __init__(self, allowed_queries):
        eos = tokenizer.eos_token
        self.allowed_queries_ids = [tokenizer(query+eos, add_special_tokens=False, return_tensors="pt").input_ids.to(device) for query in allowed_queries]
        self.starting_ids = tokenizer("Cypher output: <end_of_turn><start_of_turn>assistant\n", add_special_tokens=False, return_tensors="pt").input_ids[0].to(device)
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
        raise Exception("Genrate pattern string prompt thing is missing :/")


class MyLogitsProcessorList(list):
    def __init__(self, allowed_queries):
        super().__init__()
        self.logits_processor = MyLogitsProcessor(allowed_queries)

    def __iter__(self):
        for _ in range(1):
            yield self.logits_processor

    def __len__(self):
        return 1


schema = "Relationships without direction (:Disease)-[:OFF_LABEL_USE]-(:Drug), (:Drug)-[:OFF_LABEL_USE]-(:Disease), (:Disease)-[:ASSOCIATED_WITH]-(:GeneOrProtein), (:GeneOrProtein)-[:ASSOCIATED_WITH]-(:Disease), (:Disease)-[:PARENT_CHILD]-(:Disease), (:Disease)-[:PHENOTYPE_ABSENT]-(:EffectOrPhenotype), (:EffectOrPhenotype)-[:PHENOTYPE_ABSENT]-(:Disease), (:Disease)-[:PHENOTYPE_PRESENT]-(:EffectOrPhenotype), (:EffectOrPhenotype)-[:PHENOTYPE_PRESENT]-(:Disease), (:Disease)-[:LINKED_TO]-(:Exposure), (:Exposure)-[:LINKED_TO]-(:Disease), (:Disease)-[:CONTRAINDICATION]-(:Drug), (:Drug)-[:CONTRAINDICATION]-(:Disease), (:Disease)-[:INDICATION]-(:Drug), (:Drug)-[:INDICATION]-(:Disease), (:GeneOrProtein)-[:PPI]-(:GeneOrProtein), (:GeneOrProtein)-[:INTERACTS_WITH]-(:MolecularFunction), (:MolecularFunction)-[:INTERACTS_WITH]-(:GeneOrProtein), (:GeneOrProtein)-[:INTERACTS_WITH]-(:BiologicalProcess), (:BiologicalProcess)-[:INTERACTS_WITH]-(:GeneOrProtein), (:GeneOrProtein)-[:INTERACTS_WITH]-(:CellularComponent), (:CellularComponent)-[:INTERACTS_WITH]-(:GeneOrProtein), (:GeneOrProtein)-[:EXPRESSION_PRESENT]-(:Anatomy), (:Anatomy)-[:EXPRESSION_PRESENT]-(:GeneOrProtein), (:GeneOrProtein)-[:EXPRESSION_ABSENT]-(:Anatomy), (:Anatomy)-[:EXPRESSION_ABSENT]-(:GeneOrProtein), (:GeneOrProtein)-[:INTERACTS_WITH]-(:Pathway), (:Pathway)-[:INTERACTS_WITH]-(:GeneOrProtein), (:GeneOrProtein)-[:TARGET]-(:Drug), (:Drug)-[:TARGET]-(:GeneOrProtein), (:GeneOrProtein)-[:TRANSPORTER]-(:Drug), (:Drug)-[:TRANSPORTER]-(:GeneOrProtein), (:GeneOrProtein)-[:CARRIER]-(:Drug), (:Drug)-[:CARRIER]-(:GeneOrProtein), (:GeneOrProtein)-[:ENZYME]-(:Drug), (:Drug)-[:ENZYME]-(:GeneOrProtein), (:GeneOrProtein)-[:ASSOCIATED_WITH]-(:EffectOrPhenotype), (:EffectOrPhenotype)-[:ASSOCIATED_WITH]-(:GeneOrProtein), (:GeneOrProtein)-[:INTERACTS_WITH]-(:Exposure), (:Exposure)-[:INTERACTS_WITH]-(:GeneOrProtein), (:MolecularFunction)-[:PARENT_CHILD]-(:MolecularFunction), (:MolecularFunction)-[:INTERACTS_WITH]-(:Exposure), (:Exposure)-[:INTERACTS_WITH]-(:MolecularFunction), (:Drug)-[:SYNERGISTIC_INTERACTION]-(:Drug), (:Drug)-[:SIDE_EFFECT]-(:EffectOrPhenotype), (:EffectOrPhenotype)-[:SIDE_EFFECT]-(:Drug), (:Pathway)-[:PARENT_CHILD]-(:Pathway), (:Anatomy)-[:PARENT_CHILD]-(:Anatomy), (:EffectOrPhenotype)-[:PARENT_CHILD]-(:EffectOrPhenotype), (:BiologicalProcess)-[:PARENT_CHILD]-(:BiologicalProcess), (:BiologicalProcess)-[:INTERACTS_WITH]-(:Exposure), (:Exposure)-[:INTERACTS_WITH]-(:BiologicalProcess), (:CellularComponent)-[:PARENT_CHILD]-(:CellularComponent), (:CellularComponent)-[:INTERACTS_WITH]-(:Exposure), (:Exposure)-[:INTERACTS_WITH]-(:CellularComponent), (:Exposure)-[:PARENT_CHILD]-(:Exposure)"

instruction = "Generate Cypher statement to query a graph database. Use only the provided relationship types and properties in the schema."
base_prompt = f"<bos><start_of_turn>user\n{instruction}\nSchema: {schema}"
#base_prompt = base_prompt[:25]
print("NOT USING ENTIRE PROMPT!")

def prompt_continuation(question: str) -> str:
    rest_of_prompt = f"\nQuestion: {question} \nCypher output: <end_of_turn><start_of_turn>assistant\nMATCH"
    return rest_of_prompt

def process_base_prompt(base_prompt: str):
    tokenized_start = tokenizer("", return_tensors="pt").input_ids.to(device)
    _, new_kv_cache = model(tokenized_start, use_cache=True).values()

    tokenized_base_prompt_ids = tokenizer(base_prompt, return_tensors="pt").input_ids.to(device)
    with torch.inference_mode():
        for i in tqdm(range(tokenized_base_prompt_ids.shape[-1])):
            token = tokenized_base_prompt_ids[...,i:i+1].to(device)
            _, new_kv_cache = model(token, past_key_values=new_kv_cache).values()
    return new_kv_cache

def add_predicted_cypher(data: dict, base_prompt_cache=None) -> dict:
    beam_width = min(BEAM_WIDTH, len(data['cyphers']))
    max_new_tokens = max([len(tokenizer.tokenize(query))+10 for query in data['cyphers']])
    generation_config = GenerationConfig(num_return_sequences=beam_width, num_beams=beam_width, max_new_tokens=max_new_tokens,
                                         early_stopping=True, do_sample=False, remove_invalid_values=True)
    logits_processor_list = MyLogitsProcessorList(allowed_queries=data['cyphers'])
    if base_prompt_cache is None:
        tokenized_question = tokenizer(base_prompt+prompt_continuation(question=data['question']), return_tensors="pt").to(device)
    else:
        tokenized_question = tokenizer(prompt_continuation(question=data['question']), return_tensors="pt").to(device)
    #import copy
    #past_key_values = copy.deepcopy(base_prompt_cache)
    #past_key_values = base_prompt_cache
    outputs = model.generate(**tokenized_question, generation_config=generation_config, logits_processor=logits_processor_list,)
                             #past_key_values=past_key_values) #use another cache type to minimize memory usage (quantized? offloaded?)
                             #return_dict_in_generate=True, output_scores=True)#, past_key_values=base_prompt_cache)
    decoded_outputs = tokenizer.batch_decode(outputs, skip_special_tokens=True)
    data['top_cyphers'] = ["MATCH" + text.split("MATCH")[-1] for text in decoded_outputs]
    print(data['top_cyphers'])
    print(f"Total #cyphers:   {len(data['cyphers'])}\n Generated #cyphers: {len(data['top_cyphers'])}\n Gen valid #cyphers: {len(set(data['top_cyphers']).intersection(data['cyphers']))}")

    print("torch.cuda.memory_allocated: %fGB"%(torch.cuda.memory_allocated(0)/1024/1024/1024))
    print("torch.cuda.memory_reserved: %fGB"%(torch.cuda.memory_reserved(0)/1024/1024/1024))
    print("torch.cuda.max_memory_reserved: %fGB"%(torch.cuda.max_memory_reserved(0)/1024/1024/1024))

    return data


qa_with_train_cyphers = load_from_disk('prime-data/qa_with_train_cyphers')
#inputs = tokenizer(base_prompt, return_tensors="pt")
#prompt_cache = DynamicCache()
base_prompt_cache = None # model(**inputs, past_key_values = prompt_cache).past_key_values
qa_with_gen_cyphers = qa_with_train_cyphers.map(add_predicted_cypher)