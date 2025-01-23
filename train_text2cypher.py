from peft import LoraConfig, get_peft_model
import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)
from trl import SFTConfig, SFTTrainer




dataset = torch.load('prime-data/dataset.pt', weights_only=False)




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
    #quantization_config=bnb_config,
    torch_dtype=torch.bfloat16,
    attn_implementation="eager",
    low_cpu_mem_usage=True,
)

lora_config = LoraConfig( r=64, lora_alpha=64, target_modules=None, lora_dropout=0.05, bias="none", task_type="CAUSAL_LM", )
peft_model = get_peft_model(model=model, peft_config=lora_config)




schema = 'Relationships without direction (:Disease)-[:OFF_LABEL_USE]-(:Drug), (:Drug)-[:OFF_LABEL_USE]-(:Disease), (:Disease)-[:ASSOCIATED_WITH]-(:GeneOrProtein), (:GeneOrProtein)-[:ASSOCIATED_WITH]-(:Disease), (:Disease)-[:PARENT_CHILD]-(:Disease), (:Disease)-[:PHENOTYPE_ABSENT]-(:EffectOrPhenotype), (:EffectOrPhenotype)-[:PHENOTYPE_ABSENT]-(:Disease), (:Disease)-[:PHENOTYPE_PRESENT]-(:EffectOrPhenotype), (:EffectOrPhenotype)-[:PHENOTYPE_PRESENT]-(:Disease), (:Disease)-[:LINKED_TO]-(:Exposure), (:Exposure)-[:LINKED_TO]-(:Disease), (:Disease)-[:CONTRAINDICATION]-(:Drug), (:Drug)-[:CONTRAINDICATION]-(:Disease), (:Disease)-[:INDICATION]-(:Drug), (:Drug)-[:INDICATION]-(:Disease), (:GeneOrProtein)-[:PPI]-(:GeneOrProtein), (:GeneOrProtein)-[:INTERACTS_WITH]-(:MolecularFunction), (:MolecularFunction)-[:INTERACTS_WITH]-(:GeneOrProtein), (:GeneOrProtein)-[:INTERACTS_WITH]-(:BiologicalProcess), (:BiologicalProcess)-[:INTERACTS_WITH]-(:GeneOrProtein), (:GeneOrProtein)-[:INTERACTS_WITH]-(:CellularComponent), (:CellularComponent)-[:INTERACTS_WITH]-(:GeneOrProtein), (:GeneOrProtein)-[:EXPRESSION_PRESENT]-(:Anatomy), (:Anatomy)-[:EXPRESSION_PRESENT]-(:GeneOrProtein), (:GeneOrProtein)-[:EXPRESSION_ABSENT]-(:Anatomy), (:Anatomy)-[:EXPRESSION_ABSENT]-(:GeneOrProtein), (:GeneOrProtein)-[:INTERACTS_WITH]-(:Pathway), (:Pathway)-[:INTERACTS_WITH]-(:GeneOrProtein), (:GeneOrProtein)-[:TARGET]-(:Drug), (:Drug)-[:TARGET]-(:GeneOrProtein), (:GeneOrProtein)-[:TRANSPORTER]-(:Drug), (:Drug)-[:TRANSPORTER]-(:GeneOrProtein), (:GeneOrProtein)-[:CARRIER]-(:Drug), (:Drug)-[:CARRIER]-(:GeneOrProtein), (:GeneOrProtein)-[:ENZYME]-(:Drug), (:Drug)-[:ENZYME]-(:GeneOrProtein), (:GeneOrProtein)-[:ASSOCIATED_WITH]-(:EffectOrPhenotype), (:EffectOrPhenotype)-[:ASSOCIATED_WITH]-(:GeneOrProtein), (:GeneOrProtein)-[:INTERACTS_WITH]-(:Exposure), (:Exposure)-[:INTERACTS_WITH]-(:GeneOrProtein), (:MolecularFunction)-[:PARENT_CHILD]-(:MolecularFunction), (:MolecularFunction)-[:INTERACTS_WITH]-(:Exposure), (:Exposure)-[:INTERACTS_WITH]-(:MolecularFunction), (:Drug)-[:SYNERGISTIC_INTERACTION]-(:Drug), (:Drug)-[:SIDE_EFFECT]-(:EffectOrPhenotype), (:EffectOrPhenotype)-[:SIDE_EFFECT]-(:Drug), (:Pathway)-[:PARENT_CHILD]-(:Pathway), (:Anatomy)-[:PARENT_CHILD]-(:Anatomy), (:EffectOrPhenotype)-[:PARENT_CHILD]-(:EffectOrPhenotype), (:BiologicalProcess)-[:PARENT_CHILD]-(:BiologicalProcess), (:BiologicalProcess)-[:INTERACTS_WITH]-(:Exposure), (:Exposure)-[:INTERACTS_WITH]-(:BiologicalProcess), (:CellularComponent)-[:PARENT_CHILD]-(:CellularComponent), (:CellularComponent)-[:INTERACTS_WITH]-(:Exposure), (:Exposure)-[:INTERACTS_WITH]-(:CellularComponent), (:Exposure)-[:PARENT_CHILD]-(:Exposure)'

instruction = (
    "Generate Cypher statement to query a graph database. "
    "Use only the provided relationship types and properties in the schema. \n"
    "Schema: {schema} \n Question: {question}  \n Cypher output: "
)
label = "{cypher}"

def format_prompt_train(data):
    chat = [
        {"role": "user", "content": instruction.format(schema=schema, question=data['question'])},
        {"role": "model", "content": label.format(cypher=data['cypher'])},
    ]
    prompt = tokenizer.apply_chat_template(chat, tokenize=False)
    data['text'] = prompt[5:] #remove initial <bos> (will be added by tokenizer later)
    return data

def format_prompt_inference(data):
    chat = [
        {"role": "user", "content": instruction.format(schema=schema, question=data['question'])},
    ]
    prompt = tokenizer.apply_chat_template(chat, tokenize=False)
    data['text'] = prompt[5:]
    return data

train_data = dataset['train'].map(format_prompt_train).remove_columns(['idx', 'question', 'cypher', 'precision', 'recall'])
val_data = dataset['val'].map(format_prompt_train).remove_columns(['idx', 'question', 'cypher', 'precision', 'recall'])
test_data = dataset['test'].map(format_prompt_inference).remove_columns(['idx', 'question', 'cypher', 'precision', 'recall'])




max_seq_len = max([len(tokenizer.encode(x['text'])) for x in train_data])
max_seq_len += 100

tokenizer.padding_side = 'right'

sft_config = SFTConfig(per_device_train_batch_size=1,
                       gradient_accumulation_steps=8,
                       dataset_num_proc=16,
                       max_seq_length=max_seq_len,
                       #logging_dir="./logs",
                       num_train_epochs=1,
                       learning_rate=2e-5,
                       save_steps=5,
                       save_total_limit=1,
                       logging_steps=5,
                       output_dir="outputs",
                       optim="paged_adamw_8bit",
                       save_strategy="steps", )

trainer = SFTTrainer(model=peft_model,
                     args=sft_config,
                     processing_class=tokenizer,
                     train_dataset=train_data,
                     eval_dataset=val_data,)

trainer.train()