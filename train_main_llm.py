from peft import LoraConfig, get_peft_model
import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)
from trl import SFTConfig, SFTTrainer

def get_llm_kwargs(required_memory: int, dtype=torch.dtype) -> dict[str, any]:
    torch.cuda.empty_cache()

    gpu_memory: list[int] = []
    for i in range(torch.cuda.device_count()):
        gpu_memory.append(torch.cuda.mem_get_info(i)[0] // 1024**3)
        # Use the minimum number of GPUs to fit the LLM on.
        if sum(gpu_memory) >= required_memory:
            break

    if sum(gpu_memory) < required_memory:
        gpu_memory = []  # If not enough VRAM, use pure CPU.

    kwargs = dict(revision='main')
    if len(gpu_memory) > 0:
        kwargs['max_memory'] = {
            i: f'{memory}GiB'
            for i, memory in enumerate(gpu_memory)
        }
        kwargs['low_cpu_mem_usage'] = True
        kwargs['device_map'] = 'auto'
        kwargs['torch_dtype'] = dtype

    return kwargs

kwargs = get_llm_kwargs(required_memory=80)

#dataset = torch.load('prime-data/dataset.pt', weights_only=False)
#load new data set
dataset = torch.load('prime-data/llm2-dataset_shorter.pt', weights_only=False)


# Model
#model_name = "neo4j/text2cypher-gemma-2-9b-it-finetuned-2024v1"
model_name = "meta-llama/Llama-3.1-8B-Instruct"

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
)
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    quantization_config=bnb_config,
    torch_dtype=torch.bfloat16,
    attn_implementation="eager",
    #low_cpu_mem_usage=True,
    kwargs=kwargs,
)

lora_config = LoraConfig( r=64, lora_alpha=64, target_modules=None, lora_dropout=0.05, bias="none", task_type="CAUSAL_LM", )
peft_model = get_peft_model(model=model, peft_config=lora_config)

instruction = (
    "Given the information below, return the correct nodes for the following question: {question}\n"
    "Retrieved information:\n{info}"
    "Answer: "
)
label = "{answers}"

def format_prompt(data, train=False):
    if train:
        chat = [
            {"role": "user", "content": instruction.format(question=data['question'], info=data['info'])},
            {"role": "model", "content": label.format(answers=data['answer'])}
        ]
    else:
        chat = [
            {"role": "user", "content": instruction.format(question=data['question'], info=data['info'])},
        ]
    prompt = tokenizer.apply_chat_template(chat, tokenize=False)
    data['text'] = prompt[5:] #remove initial <bos> (will be added by tokenizer later)
    return data

train_data = dataset['train'].map(lambda x: format_prompt(x, train=True))#.remove_columns(['idx', 'question', 'cypher', 'precision', 'recall'])
print(train_data)
#val_data = dataset['val'].map(lambda x: format_prompt(x, train=True))#.remove_columns(['idx', 'question', 'cypher', 'precision', 'recall'])
#test_data = dataset['test'].map(format_prompt)#.remove_columns(['idx', 'question', 'cypher', 'precision', 'recall'])


#output dir
max_seq_len = max([len(tokenizer.encode(x['text'])) for x in train_data])
max_seq_len += 100
max_seq_len = 2_000

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
                       save_strategy="steps",
                       )

trainer = SFTTrainer(model=peft_model,
                     args=sft_config,
                     processing_class=tokenizer,
                     train_dataset=train_data,
                     #eval_dataset=val_data,
                     )

trainer.train()

text = dataset['train']['text'][0]
print("tokenize:")
tokenized = tokenizer(text, return_tensors="pt")

print("forward:")


trainer.train()