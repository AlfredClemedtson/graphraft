from unsloth import FastLanguageModel
from tqdm import tqdm
import numpy as np
from datasets import load_from_disk

max_seq_length = 11_000  # Choose any! We auto support RoPE Scaling internally!
dtype = None  # None for auto detection. Float16 for Tesla T4, V100, Bfloat16 for Ampere+
load_in_4bit = True  # Use 4bit quantization to reduce memory usage. Can be False.

model_name = "unsloth/Meta-Llama-3.1-8B-Instruct-unsloth-ft-6930"

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=model_name,
    max_seq_length=max_seq_length,
    dtype=dtype,
    load_in_4bit=load_in_4bit,
)
tokenizer.padding = True

qa_with_train_prompts = load_from_disk('prime-data/qa_with_train_prompts')

# Test inference
FastLanguageModel.for_inference(model)
prompt = qa_with_train_prompts['valid'][0]['text']
inputs = tokenizer(prompt, return_tensors='pt', padding=True, truncation=True).to("cuda")
outputs = model.generate(**inputs, min_new_tokens=10, max_new_tokens=150, num_return_sequences=1)
text = tokenizer.decode(outputs[0])
print(text)

def compute_metrics(predss: list[str], labelss: list[str]) -> dict[str, np.float32]:
    def f1(preds, labels) -> float:
        true_pos = len(set(preds).intersection(labels))
        prec = true_pos / len(preds)
        rec = true_pos / len(labels)
        return 2 * prec * rec / (prec + rec) if prec > 0 else 0

    true_poss = [len(set(preds).intersection(labels)) for preds, labels in zip(predss, labelss)]
    metrics = {
        'avg_f1' : np.mean([(f1(preds, labels)) for preds, labels in zip(predss, labelss)]),
        'avg_prec' : np.mean([true_pos/len(preds) for preds, true_pos in zip(predss, true_poss)]),
        'avg_rec' : np.mean([true_pos/len(labels) for labels, true_pos in zip(labelss, true_poss)]),
        'avg_hits_at_1' : np.mean([1 if true_pos > 0 else 0 for true_pos in true_poss]),
        'avg_num_preds' : np.mean([len(preds) for preds in predss])
    }
    return metrics

predss = []; labelss = []
with tqdm() as pbar:
    for data in qa_with_train_prompts['valid']:
        answers = data['answer']
        prompt = data['text']
        inputs = tokenizer(prompt, return_tensors='pt').to("cuda")
        outputs = model.generate(**inputs, min_new_tokens=10, max_new_tokens=150, num_return_sequences=1)
        text = tokenizer.decode(outputs[0])
        try:
            preds = text.split("<|eot_id|><|start_header_id|>model<|end_header_id|>\n\n")[1].split("<|eot_id|>")[0].split('|')
            predss.append(preds)
        except:
            predss.append([])
        labelss.append(answers.split('|'))
        metrics = compute_metrics(predss, labelss)

        pbar.set_description(
            f"Avg f1: {metrics['avg_f1']:.3f}, Avg prec: {metrics['avg_prec']:.3f}, Avg rec: {metrics['avg_rec']:.3f}, "
            f"Hits@1: {metrics['avg_hits_at_1']:.3f}, Avg preds: {metrics['avg_num_preds']}")
        pbar.update(1)


metrics = compute_metrics(predss, labelss)
print(f"Avg f1: {metrics['avg_f1']:.3f}, Avg prec: {metrics['avg_prec']:.3f}, Avg rec: {metrics['avg_rec']:.3f} "
      f"Hits@1: {metrics['avg_hits_at_1']:.3f}, Avg preds: {metrics['avg_num_preds']}")