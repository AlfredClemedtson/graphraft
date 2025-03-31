from collections import defaultdict
import json
from openai import OpenAI
from neo4j import Driver

#CREATE FULLTEXT INDEX _Entity_FullTextIndex FOR (n:_Entity_) ON EACH [n.name]

class NER:
    def __init__(self, dataset_name, model='gpt-4o-mini', openai_api_key: str = None):
        self.model = model
        self.openai_api_key = openai_api_key

        ner_instructions = json.load(open(f"{dataset_name}-data/ner_instructions.json"))
        self.system_instruction = ner_instructions['system_instruction']
        self.multi_shot_examples = ner_instructions['multi_shot_examples']
        self.labels = ner_instructions['labels']

        if model=='gpt-4o-mini':
            self.pipeline = ...
        else:
            import transformers, torch
            self.pipeline = transformers.pipeline(
                "text-generation",
                model="model",#"meta-llama/Meta-Llama-3.1-8B-Instruct",
                model_kwargs={"torch_dtype": torch.bfloat16},
                device_map="auto",
            )


    def find_source_nodes(self, question: str, driver: Driver):
        named_entities = self._identify_unlabeled_entities(question) if self.labels is None else self._identify_labeled_entities(question)
        print("Identified by LLM: ", named_entities)

        matched_nodes = self._match_labeled_entities(driver=driver, entities=named_entities)
        return matched_nodes


    def _ask_ai(self, question: str, model='gpt-4o-mini'):
        user_model_correspondence = [({"role": "user", "content": f"Q:\"{multi_shot_example['question']}\""},
                                      {"role": "assistant", "content": f"A:{multi_shot_example['answer']}"}) for
                                     multi_shot_example in self.multi_shot_examples]
        user_model_correspondence = [x for xs in user_model_correspondence for x in xs]  # flatten
        messages = [
            {"role": "system", "content": self.system_instruction},
            *user_model_correspondence,
            {"role": "user", "content": f"Q:\"{question}"},
        ]
        if model=='gpt-4o-mini':
            client = OpenAI(api_key=self.openai_api_key)
            completion = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=messages
            )
            response = completion.choices[0].message.content.lstrip('A').lstrip(':')

        elif model=="llama-3.1-8b-instruct":
            outputs = self.pipeline(
                messages,
                max_new_tokens=256,
            )
            response = outputs[0]["generated_text"][-1].lstrip('A').lstrip(':')
        else:
            raise ValueError(f"Unrecognized model: {model}")

        return response


    def _identify_unlabeled_entities(self, question: str) -> list[tuple[str, str]]:
        response = self._ask_ai(question)
        entities = response.lstrip('A').lstrip(':').split('|')
        labeled_entities = [('_Entity_', entity) for entity in entities]
        return labeled_entities


    def _identify_labeled_entities(self, question: str) -> list[tuple[str, str]]:
        response = self._ask_ai(question)

        idx0s, idx1s = [], []
        for label in self.labels:
            label_str = label + ':'
            if label_str not in response:
                continue
            idx0s.append(response.index(label_str))
            idx1s.append(response.index(label_str) + len(label_str))
        idx0s, idx1s = sorted(idx0s), sorted(idx1s)

        labeled_entities = []
        for i in range(len(idx0s)):
            label = response[idx0s[i]:idx1s[i]-1]
            value = response[idx1s[i]:idx0s[i+1]] if i < len(idx0s) - 1 else response[idx1s[i]:]
            labeled_entities.append((label, value.strip()))
        return labeled_entities


    def _match_labeled_entities(self, driver: Driver, entities: list[tuple[str, str]], k=100):
        node_names = []
        for label, string in entities:
            try:
                res = driver.execute_query(FULL_TEXT_QUERY, parameters_={'string': string, 'fullTextIndex': label+'FullTextIndex'})
                node_names.extend([rec['name'] for rec in res.records])
            except Exception as e:
                res = None
            if res is None or len(res.records) == 0:
                res = driver.execute_query(vector_sim_query, parameters_={'string': string, 'k': k,
                                                                         'api_key': self.openai_api_key,
                                                                         'vectorIndex': label+'NameEmbedding'})
                node_names.extend([rec['name'] for rec in res.records])
        return node_names

from typing import LiteralString
FULL_TEXT_QUERY: LiteralString = """CALL db.index.fulltext.queryNodes($fullTextIndex, $string) YIELD node, score
                                    WITH node.name AS name
                                    WHERE normalize(lower(name)) = normalize(lower($string))
                                    RETURN name"""


vector_sim_query: LiteralString = """WITH genai.vector.encode($string, 'OpenAI', { token: $api_key }) AS embedding
                                     CALL db.index.vector.queryNodes($vectorIndex, $k, embedding) YIELD node
                                     RETURN node.name AS name
                                     LIMIT 1"""