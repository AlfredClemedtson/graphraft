import re
from peft import PeftModel
import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig, LlamaTokenizer,
)
from datasets import DatasetDict, load_from_disk

def cypher2path(cypher_query: str) -> list[tuple[str, str, str, str]]:
    path = re.findall(r"(?:\(|-\[)(x|r)(\d):([^ \)\]]+)(?: \{name: \"(.+)\"\})?(?:\)|\]-)", cypher_query)
    return path

def block2cypher(x_r: str, num: str, label_or_type: str, name: str) -> str:
    if x_r == 'x':
        prop_string = f" {{name: \"{name}\"}}" if name != '' else ""
        return f"(x{num}:{label_or_type}{prop_string})"
    elif x_r == 'r':
        return f"-[r{num}:{label_or_type}]-"

def path2cypher(path: list[tuple[str, str, str, str]]) -> str:
    query = "MATCH "
    for x_r, num, labelOrType, name in path:
        if x_r == 'x' or x_r == 'r':
            query += block2cypher(x_r, num, labelOrType, name)
        elif x_r == '':
            query += f" RETURN x{num}.name as name"
    return query


def get_options(current_path: list[tuple[str, str, str, str]],
                all_paths: list[list[tuple[str, str, str, str]]]) -> set[str]:
    options = set()
    for path in all_paths:
        for i, (x_r, num, labelOrType, name) in enumerate(path):
            if i == len(current_path):
                options.add(block2cypher(x_r, num, labelOrType, name))
                break
            if x_r != current_path[i][0] or str(num) != current_path[i][1] or labelOrType != current_path[i][
                2] or name != current_path[i][3]:
                break  # path does not match
    return options


def load_model(model_dir: str):
    # Model
    model_name = "neo4j/text2cypher-gemma-2-9b-it-finetuned-2024v1"
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        torch_dtype=torch.bfloat16,
        attn_implementation="eager",
        low_cpu_mem_usage=True,
    ).to('cuda')

    peft_model = PeftModel.from_pretrained(model, model_dir,
                                           torch_dtype=torch.bfloat16,
                                           is_trainable=False)
    return peft_model, tokenizer

def format_tokenize_and_score_prompt_and_cypher(model: PeftModel,
                                                tokenizer: LlamaTokenizer,
                                                question: str,
                                                cypher: str) -> float:
    schema = 'Relationships without direction (:Disease)-[:OFF_LABEL_USE]-(:Drug), (:Drug)-[:OFF_LABEL_USE]-(:Disease), (:Disease)-[:ASSOCIATED_WITH]-(:GeneOrProtein), (:GeneOrProtein)-[:ASSOCIATED_WITH]-(:Disease), (:Disease)-[:PARENT_CHILD]-(:Disease), (:Disease)-[:PHENOTYPE_ABSENT]-(:EffectOrPhenotype), (:EffectOrPhenotype)-[:PHENOTYPE_ABSENT]-(:Disease), (:Disease)-[:PHENOTYPE_PRESENT]-(:EffectOrPhenotype), (:EffectOrPhenotype)-[:PHENOTYPE_PRESENT]-(:Disease), (:Disease)-[:LINKED_TO]-(:Exposure), (:Exposure)-[:LINKED_TO]-(:Disease), (:Disease)-[:CONTRAINDICATION]-(:Drug), (:Drug)-[:CONTRAINDICATION]-(:Disease), (:Disease)-[:INDICATION]-(:Drug), (:Drug)-[:INDICATION]-(:Disease), (:GeneOrProtein)-[:PPI]-(:GeneOrProtein), (:GeneOrProtein)-[:INTERACTS_WITH]-(:MolecularFunction), (:MolecularFunction)-[:INTERACTS_WITH]-(:GeneOrProtein), (:GeneOrProtein)-[:INTERACTS_WITH]-(:BiologicalProcess), (:BiologicalProcess)-[:INTERACTS_WITH]-(:GeneOrProtein), (:GeneOrProtein)-[:INTERACTS_WITH]-(:CellularComponent), (:CellularComponent)-[:INTERACTS_WITH]-(:GeneOrProtein), (:GeneOrProtein)-[:EXPRESSION_PRESENT]-(:Anatomy), (:Anatomy)-[:EXPRESSION_PRESENT]-(:GeneOrProtein), (:GeneOrProtein)-[:EXPRESSION_ABSENT]-(:Anatomy), (:Anatomy)-[:EXPRESSION_ABSENT]-(:GeneOrProtein), (:GeneOrProtein)-[:INTERACTS_WITH]-(:Pathway), (:Pathway)-[:INTERACTS_WITH]-(:GeneOrProtein), (:GeneOrProtein)-[:TARGET]-(:Drug), (:Drug)-[:TARGET]-(:GeneOrProtein), (:GeneOrProtein)-[:TRANSPORTER]-(:Drug), (:Drug)-[:TRANSPORTER]-(:GeneOrProtein), (:GeneOrProtein)-[:CARRIER]-(:Drug), (:Drug)-[:CARRIER]-(:GeneOrProtein), (:GeneOrProtein)-[:ENZYME]-(:Drug), (:Drug)-[:ENZYME]-(:GeneOrProtein), (:GeneOrProtein)-[:ASSOCIATED_WITH]-(:EffectOrPhenotype), (:EffectOrPhenotype)-[:ASSOCIATED_WITH]-(:GeneOrProtein), (:GeneOrProtein)-[:INTERACTS_WITH]-(:Exposure), (:Exposure)-[:INTERACTS_WITH]-(:GeneOrProtein), (:MolecularFunction)-[:PARENT_CHILD]-(:MolecularFunction), (:MolecularFunction)-[:INTERACTS_WITH]-(:Exposure), (:Exposure)-[:INTERACTS_WITH]-(:MolecularFunction), (:Drug)-[:SYNERGISTIC_INTERACTION]-(:Drug), (:Drug)-[:SIDE_EFFECT]-(:EffectOrPhenotype), (:EffectOrPhenotype)-[:SIDE_EFFECT]-(:Drug), (:Pathway)-[:PARENT_CHILD]-(:Pathway), (:Anatomy)-[:PARENT_CHILD]-(:Anatomy), (:EffectOrPhenotype)-[:PARENT_CHILD]-(:EffectOrPhenotype), (:BiologicalProcess)-[:PARENT_CHILD]-(:BiologicalProcess), (:BiologicalProcess)-[:INTERACTS_WITH]-(:Exposure), (:Exposure)-[:INTERACTS_WITH]-(:BiologicalProcess), (:CellularComponent)-[:PARENT_CHILD]-(:CellularComponent), (:CellularComponent)-[:INTERACTS_WITH]-(:Exposure), (:Exposure)-[:INTERACTS_WITH]-(:CellularComponent), (:Exposure)-[:PARENT_CHILD]-(:Exposure)'

    instruction = (
        "Generate Cypher statement to query a graph database. "
        "Use only the provided relationship types and properties in the schema. \n"
        "Schema: {schema} \n Question: {question}  \n Cypher output: "
    )
    label = "{cypher}"

    chat = [
        {"role": "user", "content": instruction.format(schema=schema, question=question)},
        {"role": "model", "content": label.format(cypher=cypher)},
    ]
    input_ids = tokenizer.apply_chat_template(chat, tokenize=True, return_tensors='pt').to('cuda')
    output = model(input_ids, labels=input_ids)
    loss = output.loss.cpu().detach().numpy()
    return -loss.item()

    #prod( p(t_i | t_(<i) ))
    #"What is a good medication for my headache? (x {'name' : 'headache'})"
    #"What is a good medication for my headache? (x {'name' : 'leg'})"

def generate_cypher(model: PeftModel, tokenizer: LlamaTokenizer, question: str, all_paths: list, beam_width: int):
    top_queries = [f"MATCH {part}" for part in get_options([], all_paths)]

    for i in range(1, 5):
        optionss = [get_options(cypher2path(query), all_paths) for query in top_queries]
        queries = [query + option for query, options in zip(top_queries, optionss) for option in options]
        x = [(query, format_tokenize_and_score_prompt_and_cypher(model, tokenizer, question, query)) for query in queries]
        top_queries, probs = zip(*sorted(x, key=lambda x: x[1], reverse=True)[:beam_width])
        if i == 2:  # possibly stop here
            one_hop_queries, one_hop_probs = top_queries, probs
    top_queries += one_hop_queries
    probs += one_hop_probs
    top_queries, probs = zip(*sorted(zip(top_queries, probs), key=lambda x: x[1], reverse=True))
    return top_queries, probs

def add_topk_cyphers(data: dict, model: PeftModel, tokenizer: LlamaTokenizer, beam_width: int):
    question = data['question']
    all_paths = [cypher2path(cypher) for cypher in data['cyphers']]
    top_queries, probs = generate_cypher(model, tokenizer, question, all_paths, beam_width=beam_width)
    data['cypher_preds'] = top_queries
    # for query, prob in zip(top_queries, probs):
    #     print(prob, query)
    # print()
    return data


beam_width = 5
model_dir = './outputs/checkpoint-667'
model, tokenizer = load_model(model_dir=model_dir)

all_cyphers_dataset = load_from_disk('prime-data/qa_with_train_cyphers')
pred_cyphers_dataset_valid = all_cyphers_dataset['valid'].map(lambda x: add_topk_cyphers(model=model, tokenizer=tokenizer, beam_width=beam_width, data=x))

pred_cyphers_dataset = DatasetDict({'valid': pred_cyphers_dataset_valid})
pred_cyphers_dataset.save_to_disk('prime_data/qa_with_pred_cyphers')