import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM, StaticCache, CacheConfig, BitsAndBytesConfig
from torch.nn.functional import log_softmax
from datasets import load_from_disk
from tqdm import tqdm

BEAM_WIDTH = 5
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#device = 'cpu'

# Model
#model_name = "neo4j/text2cypher-gemma-2-9b-it-finetuned-2024v1"
model_name = "meta-llama/Llama-3.2-1B-Instruct"

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
)
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    #quantization_config=bnb_config,
    torch_dtype=torch.bfloat16,
    attn_implementation="eager",
    low_cpu_mem_usage=True,
).to(device)


class Path:
    def __init__(self, ids, losses, cache):
        self.ids = ids
        self.text = tokenizer.decode(ids[0, :])
        self.losses = losses
        self.cache = cache

    def loss(self) -> float:
        return np.mean(self.losses)

    def new_loss_lower_bound(self) -> float:
        return np.mean(self.losses + [0])

    def is_finished(self):
        return self.text.strip().endswith('RETURN')

    def allowed_next_ids(self, all_allowed_queries):
        token_ids = set()
        for query in all_allowed_queries:
            if query.startswith(self.text):
                allowed_continuation = query[len(self.text):]
                if allowed_continuation == "":
                    continue
                token_ids.add(tokenizer(allowed_continuation, add_special_tokens=False).input_ids[0])
        return token_ids

    def plus_one_token(self, all_allowed_queries):  # explore all possible steps and add to top list if loss is good enough
        last_id = self.ids[..., -1:].to(device)
        logits, new_kv_cache = model(last_id, past_key_values=self.cache, use_cache=True).values()
        log_probs = log_softmax(logits, dim=-1)

        for next_token_id in self.allowed_next_ids(all_allowed_queries=all_allowed_queries):
            new_ids = torch.cat((self.ids, torch.tensor([[next_token_id]])), dim=-1)
            new_losses = self.losses + [-log_probs[..., next_token_id].item()]
            yield Path(new_ids, new_losses, new_kv_cache)

class Bundle:
    def __init__(self, beam_width: int):
        self.beam_width = beam_width
        self.paths: list[Path] = []

    def worst(self) -> float:
        if len(self.paths) < self.beam_width:
            return float('inf')  # if not full
        else:
            return self.paths[-1].loss()

    def _inplace_sort(self):
        self.paths.sort(key=lambda path: path.loss(), reverse=False)

    def push(self, new_path: Path):
        # Worse
        if new_path.loss() > self.worst():
            return  # not good enough :/
        # Already exists
        for i, path in enumerate(self.paths):
            if path.text == new_path.text:
                if new_path.loss() >= path.loss():
                    return  # do nothing, the new one is worse
                else:
                    self.paths[i] = new_path
                    self._inplace_sort()
                    return  # keep the best score, then re-rank
        # Already seen (but not there atm, e.g. ['D', 'isease'] vs ['Disease'])
        ...  # TODO handle if exists, or already seen?

        # All fine, add it.
        self.paths.append(new_path)
        self._inplace_sort()
        self.paths = self.paths[:self.beam_width]
        return

def plus_one_token(beam_width: int, all_allowed_queries: list[str], bundle: Bundle, finished_paths: list[Path]) -> Bundle:
    new_bundle = Bundle(beam_width=beam_width)
    for path in bundle.paths:
        if path.new_loss_lower_bound() > new_bundle.worst():  # cannot be good, don't even try
            continue
        for new_path in path.plus_one_token(all_allowed_queries=all_allowed_queries):
            if new_path.is_finished():  # it is finished, save but don't continue on this
                # del new_path.cache # maybe unnecessary
                finished_paths.append(new_path)
            else:
                new_bundle.push(new_path)  # try to add it, will be discarded if not good enough
        # del path.cache # maybe unnecessary
    return new_bundle

def run_beam_search(beam_width: int, all_allowed_queries: list[str], init_cache: StaticCache = None) -> list[Path]:
    if init_cache is None:
        init_cache = StaticCache(CacheConfig())
    init_ids = tokenizer('MATCH', add_special_tokens=False, return_tensors='pt').input_ids
    init_path = Path(ids=init_ids, losses=[], cache=init_cache)
    bundle = Bundle(beam_width=beam_width)
    bundle.push(init_path)
    finished_paths = []

    while bundle.paths:  # when not empty (will never stop if allowed to continue!)
        for path in bundle.paths:
            print(path.text)
        print(torch.cuda.max_memory_allocated(device=device))
        bundle = plus_one_token(beam_width=beam_width, all_allowed_queries=all_allowed_queries, bundle=bundle,
                                finished_paths=finished_paths)  # send reference of finished_paths (should be modified)

    unique_texts = set()
    unique_paths = []
    for path in sorted(finished_paths, key=lambda path: path.loss(), reverse=False):  # sort by loss, lower is better
        if path.text in unique_texts:
            unique_paths.append(path)
        unique_texts.add(path.text)
    return unique_paths

# def format_and_tokenize_prompt(tokenizer: AutoTokenizer, question: str) -> torch.tensor:
#     schema = 'Relationships without direction (:Disease)-[:OFF_LABEL_USE]-(:Drug), (:Drug)-[:OFF_LABEL_USE]-(:Disease), (:Disease)-[:ASSOCIATED_WITH]-(:GeneOrProtein), (:GeneOrProtein)-[:ASSOCIATED_WITH]-(:Disease), (:Disease)-[:PARENT_CHILD]-(:Disease), (:Disease)-[:PHENOTYPE_ABSENT]-(:EffectOrPhenotype), (:EffectOrPhenotype)-[:PHENOTYPE_ABSENT]-(:Disease), (:Disease)-[:PHENOTYPE_PRESENT]-(:EffectOrPhenotype), (:EffectOrPhenotype)-[:PHENOTYPE_PRESENT]-(:Disease), (:Disease)-[:LINKED_TO]-(:Exposure), (:Exposure)-[:LINKED_TO]-(:Disease), (:Disease)-[:CONTRAINDICATION]-(:Drug), (:Drug)-[:CONTRAINDICATION]-(:Disease), (:Disease)-[:INDICATION]-(:Drug), (:Drug)-[:INDICATION]-(:Disease), (:GeneOrProtein)-[:PPI]-(:GeneOrProtein), (:GeneOrProtein)-[:INTERACTS_WITH]-(:MolecularFunction), (:MolecularFunction)-[:INTERACTS_WITH]-(:GeneOrProtein), (:GeneOrProtein)-[:INTERACTS_WITH]-(:BiologicalProcess), (:BiologicalProcess)-[:INTERACTS_WITH]-(:GeneOrProtein), (:GeneOrProtein)-[:INTERACTS_WITH]-(:CellularComponent), (:CellularComponent)-[:INTERACTS_WITH]-(:GeneOrProtein), (:GeneOrProtein)-[:EXPRESSION_PRESENT]-(:Anatomy), (:Anatomy)-[:EXPRESSION_PRESENT]-(:GeneOrProtein), (:GeneOrProtein)-[:EXPRESSION_ABSENT]-(:Anatomy), (:Anatomy)-[:EXPRESSION_ABSENT]-(:GeneOrProtein), (:GeneOrProtein)-[:INTERACTS_WITH]-(:Pathway), (:Pathway)-[:INTERACTS_WITH]-(:GeneOrProtein), (:GeneOrProtein)-[:TARGET]-(:Drug), (:Drug)-[:TARGET]-(:GeneOrProtein), (:GeneOrProtein)-[:TRANSPORTER]-(:Drug), (:Drug)-[:TRANSPORTER]-(:GeneOrProtein), (:GeneOrProtein)-[:CARRIER]-(:Drug), (:Drug)-[:CARRIER]-(:GeneOrProtein), (:GeneOrProtein)-[:ENZYME]-(:Drug), (:Drug)-[:ENZYME]-(:GeneOrProtein), (:GeneOrProtein)-[:ASSOCIATED_WITH]-(:EffectOrPhenotype), (:EffectOrPhenotype)-[:ASSOCIATED_WITH]-(:GeneOrProtein), (:GeneOrProtein)-[:INTERACTS_WITH]-(:Exposure), (:Exposure)-[:INTERACTS_WITH]-(:GeneOrProtein), (:MolecularFunction)-[:PARENT_CHILD]-(:MolecularFunction), (:MolecularFunction)-[:INTERACTS_WITH]-(:Exposure), (:Exposure)-[:INTERACTS_WITH]-(:MolecularFunction), (:Drug)-[:SYNERGISTIC_INTERACTION]-(:Drug), (:Drug)-[:SIDE_EFFECT]-(:EffectOrPhenotype), (:EffectOrPhenotype)-[:SIDE_EFFECT]-(:Drug), (:Pathway)-[:PARENT_CHILD]-(:Pathway), (:Anatomy)-[:PARENT_CHILD]-(:Anatomy), (:EffectOrPhenotype)-[:PARENT_CHILD]-(:EffectOrPhenotype), (:BiologicalProcess)-[:PARENT_CHILD]-(:BiologicalProcess), (:BiologicalProcess)-[:INTERACTS_WITH]-(:Exposure), (:Exposure)-[:INTERACTS_WITH]-(:BiologicalProcess), (:CellularComponent)-[:PARENT_CHILD]-(:CellularComponent), (:CellularComponent)-[:INTERACTS_WITH]-(:Exposure), (:Exposure)-[:INTERACTS_WITH]-(:CellularComponent), (:Exposure)-[:PARENT_CHILD]-(:Exposure)'
#
#     instruction = (
#         "Generate Cypher statement to query a graph database. "
#         "Use only the provided relationship types and properties in the schema. \n"
#         "Schema: {schema} \n Question: {question}  \n Cypher output: "
#     )
#     #label = "{cypher}"
#     chat = [
#         {"role": "user", "content": instruction.format(schema=schema, question=question)},
#         #{"role": "model", "content": label.format(cypher=cypher)},
#     ]
#     input_ids = tokenizer.apply_chat_template(chat, tokenize=True, return_tensors='pt', return_dict=True).input_ids
#     return input_ids
schema = "Relationships without direction (:Disease)-[:OFF_LABEL_USE]-(:Drug), (:Drug)-[:OFF_LABEL_USE]-(:Disease), (:Disease)-[:ASSOCIATED_WITH]-(:GeneOrProtein), (:GeneOrProtein)-[:ASSOCIATED_WITH]-(:Disease), (:Disease)-[:PARENT_CHILD]-(:Disease), (:Disease)-[:PHENOTYPE_ABSENT]-(:EffectOrPhenotype), (:EffectOrPhenotype)-[:PHENOTYPE_ABSENT]-(:Disease), (:Disease)-[:PHENOTYPE_PRESENT]-(:EffectOrPhenotype), (:EffectOrPhenotype)-[:PHENOTYPE_PRESENT]-(:Disease), (:Disease)-[:LINKED_TO]-(:Exposure), (:Exposure)-[:LINKED_TO]-(:Disease), (:Disease)-[:CONTRAINDICATION]-(:Drug), (:Drug)-[:CONTRAINDICATION]-(:Disease), (:Disease)-[:INDICATION]-(:Drug), (:Drug)-[:INDICATION]-(:Disease), (:GeneOrProtein)-[:PPI]-(:GeneOrProtein), (:GeneOrProtein)-[:INTERACTS_WITH]-(:MolecularFunction), (:MolecularFunction)-[:INTERACTS_WITH]-(:GeneOrProtein), (:GeneOrProtein)-[:INTERACTS_WITH]-(:BiologicalProcess), (:BiologicalProcess)-[:INTERACTS_WITH]-(:GeneOrProtein), (:GeneOrProtein)-[:INTERACTS_WITH]-(:CellularComponent), (:CellularComponent)-[:INTERACTS_WITH]-(:GeneOrProtein), (:GeneOrProtein)-[:EXPRESSION_PRESENT]-(:Anatomy), (:Anatomy)-[:EXPRESSION_PRESENT]-(:GeneOrProtein), (:GeneOrProtein)-[:EXPRESSION_ABSENT]-(:Anatomy), (:Anatomy)-[:EXPRESSION_ABSENT]-(:GeneOrProtein), (:GeneOrProtein)-[:INTERACTS_WITH]-(:Pathway), (:Pathway)-[:INTERACTS_WITH]-(:GeneOrProtein), (:GeneOrProtein)-[:TARGET]-(:Drug), (:Drug)-[:TARGET]-(:GeneOrProtein), (:GeneOrProtein)-[:TRANSPORTER]-(:Drug), (:Drug)-[:TRANSPORTER]-(:GeneOrProtein), (:GeneOrProtein)-[:CARRIER]-(:Drug), (:Drug)-[:CARRIER]-(:GeneOrProtein), (:GeneOrProtein)-[:ENZYME]-(:Drug), (:Drug)-[:ENZYME]-(:GeneOrProtein), (:GeneOrProtein)-[:ASSOCIATED_WITH]-(:EffectOrPhenotype), (:EffectOrPhenotype)-[:ASSOCIATED_WITH]-(:GeneOrProtein), (:GeneOrProtein)-[:INTERACTS_WITH]-(:Exposure), (:Exposure)-[:INTERACTS_WITH]-(:GeneOrProtein), (:MolecularFunction)-[:PARENT_CHILD]-(:MolecularFunction), (:MolecularFunction)-[:INTERACTS_WITH]-(:Exposure), (:Exposure)-[:INTERACTS_WITH]-(:MolecularFunction), (:Drug)-[:SYNERGISTIC_INTERACTION]-(:Drug), (:Drug)-[:SIDE_EFFECT]-(:EffectOrPhenotype), (:EffectOrPhenotype)-[:SIDE_EFFECT]-(:Drug), (:Pathway)-[:PARENT_CHILD]-(:Pathway), (:Anatomy)-[:PARENT_CHILD]-(:Anatomy), (:EffectOrPhenotype)-[:PARENT_CHILD]-(:EffectOrPhenotype), (:BiologicalProcess)-[:PARENT_CHILD]-(:BiologicalProcess), (:BiologicalProcess)-[:INTERACTS_WITH]-(:Exposure), (:Exposure)-[:INTERACTS_WITH]-(:BiologicalProcess), (:CellularComponent)-[:PARENT_CHILD]-(:CellularComponent), (:CellularComponent)-[:INTERACTS_WITH]-(:Exposure), (:Exposure)-[:INTERACTS_WITH]-(:CellularComponent), (:Exposure)-[:PARENT_CHILD]-(:Exposure)"

instruction = "Generate Cypher statement to query a graph database. Use only the provided relationship types and properties in the schema."
base_prompt = f"<bos><start_of_turn>user\n{instruction}\nSchema: {schema}"

def prompt_continuation(question: str) -> str:
    rest_of_prompt = f"\nQuestion: {question} \nCypher output: <end_of_turn>\n"
    return rest_of_prompt

# def process_tokenized_prompt(prompt_ids: torch.tensor) -> StaticCache:
#     #TODO: Can be made more efficient by only processing the schema once!
#
#     tokenized_start = tokenizer("", return_tensors="pt").input_ids.to(device)
#     _, new_kv_cache = model(tokenized_start, use_cache=True).values()
#     with torch.inference_mode():
#         for i in range(prompt_ids.shape[-1]):
#             token = prompt_ids[...,i:i+1].to(device)
#             _, new_kv_cache = model(token, past_key_values=new_kv_cache, use_cache=True).values()
#     return new_kv_cache

def process_base_prompt(base_prompt: str) -> StaticCache:
    tokenized_start = tokenizer("", return_tensors="pt").input_ids.to(device)
    _, new_kv_cache = model(tokenized_start, use_cache=True).values()

    tokenized_base_prompt_ids = tokenizer(base_prompt, return_tensors="pt").input_ids.to(device)
    with torch.inference_mode():
        for i in tqdm(range(tokenized_base_prompt_ids.shape[-1])):
            token = tokenized_base_prompt_ids[...,i:i+1].to(device)
            _, new_kv_cache = model(token, past_key_values=new_kv_cache).values()
    return new_kv_cache

def process_prompt_continuation(base_prompt_cache: StaticCache, prompt_continuation_ids: torch.tensor) -> StaticCache:
    new_kv_cache = base_prompt_cache#.copy() #necessary? sufficient (deepcopy)?
    with torch.inference_mode():
        for i in range(prompt_continuation_ids.shape[-1]):
            token = prompt_continuation_ids[...,i:i+1].to(device)
            _, new_kv_cache = model(token, past_key_values=new_kv_cache).values()
    return new_kv_cache

def add_predicted_cypher(base_prompt_cache: StaticCache, data: dict) -> dict:
    question = data['question']
    all_allowed_queries = data['cyphers']
    prompt_ids = tokenizer(prompt_continuation(question=question), return_tensors="pt").input_ids.to(device)
    init_cache = process_prompt_continuation(base_prompt_cache, prompt_ids)#process_tokenized_prompt(prompt_ids)
    output_paths = run_beam_search(beam_width=BEAM_WIDTH, all_allowed_queries=all_allowed_queries, init_cache=init_cache)
    data['top_cyphers'] = [path.text for path in output_paths[:BEAM_WIDTH]]
    return data


qa_with_train_cyphers = load_from_disk('prime-data/qa_with_train_cyphers')
base_prompt_cache = process_base_prompt(base_prompt=base_prompt)

#new_data = add_predicted_cypher(base_prompt_cache, qa_with_train_cyphers['train'][0])
#print(new_data['top_cyphers'])
qa_with_gen_cyphers = qa_with_train_cyphers.map(lambda x: add_predicted_cypher(base_prompt_cache, x))