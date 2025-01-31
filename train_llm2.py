from unsloth import FastLanguageModel
import time
from datasets import load_from_disk

max_seq_length = 11_000  # Choose any! We auto support RoPE Scaling internally!
dtype = None  # None for auto detection. Float16 for Tesla T4, V100, Bfloat16 for Ampere+
load_in_4bit = True  # Use 4bit quantization to reduce memory usage. Can be False.

# model_name = "unsloth/Llama-3.2-1B-Instruct"
model_name = "unsloth/Meta-Llama-3.1-8B-Instruct"
# model_name = "unsloth/Meta-Llama-3.1-8B-Instruct-unsloth-ft-6930"

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=model_name,
    max_seq_length=max_seq_length,
    dtype=dtype,
    load_in_4bit=load_in_4bit,
    # token = "hf_...", # use one if using gated models like meta-llama/Llama-2-7b-hf
)
tokenizer.padding = True

model = FastLanguageModel.get_peft_model(
    model,
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


def format_prompt(data, train=False):
    instruction = (
        "Given the information below, return the correct nodes for the following question: {question}\n"
        "Retrieved information:\n{info}"
        "Answer: "
    )
    response = "{answers}"
    chat = [{"role": "user", "content": instruction.format(question=data['question'], info=data['info'])}]
    if train:
        chat.append({"role": "model", "content": response.format(answers=data['answer'])})
    prompt = tokenizer.apply_chat_template(chat, tokenize=False, padding=True, max_length=max_seq_length)
    data['text'] = prompt #do check if still adds <bos> here and at tokenizer
    return data


qa_with_train_prompts = load_from_disk('prime-data/qa_with_train_prompts')
# qa_with_train_prompts = qa_with_train_prompts.filter(lambda _,i: i%100==0, with_indices=True)
qa_with_train_prompts = qa_with_train_prompts.map(lambda x: format_prompt(x, train=True), num_proc=8)

from trl import SFTTrainer, SFTConfig
from unsloth import is_bfloat16_supported

trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=qa_with_train_prompts['train'],
    # eval_dataset = qa_with_train_prompts['valid'],
    dataset_text_field="text",
    max_seq_length=max_seq_length,
    dataset_num_proc=8,
    packing=False,  # Can make training 5x faster for short sequences.
    args=SFTConfig(
        per_device_train_batch_size=1,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=4,
        warmup_steps=5,
        num_train_epochs=1,  # Set this for 1 full training run.
        # max_steps = 500,
        learning_rate=2e-4,
        fp16=not is_bfloat16_supported(),
        bf16=is_bfloat16_supported(),
        logging_steps=1,
        optim="adamw_8bit",
        weight_decay=0.01,
        lr_scheduler_type="linear",
        seed=3407,
        output_dir="outputs",
        report_to="none",  # Use this for WandB etc
        # eval_strategy='steps',
        # eval_steps=20,
        save_strategy='steps',
        save_steps=500,
    ),
)

#Train
from unsloth.chat_templates import train_on_responses_only
trainer = train_on_responses_only(
    trainer,
    instruction_part="<|start_header_id|>user<|end_header_id|>\n\n",
    response_part="<|start_header_id|>model<|end_header_id|>\n\n",
)
trainer_stats = trainer.train()

# Save
num = int(time.time() / 10) % 10_000
new_model_name = f"{model_name}-unsloth-ft-{num}"
model.save_pretrained(new_model_name)
tokenizer.save_pretrained(new_model_name)


