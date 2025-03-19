from collections import defaultdict

import torch
import transformers
from openai import OpenAI
from neo4j import Driver

PRIME_MULTISHOT_EXAMPLES = [
            {"question" : "Which anatomical structures lack the expression of genes or proteins involved in the interaction with the fucose metabolism pathway?", "answer" : "fucose metabolism"},
            {"question" : "What liquid drugs target the A2M gene/protein and bind to the PDGFR-beta receptor?", "answer" : "A2M gene/protein|PDGFR-beta receptor"},
            {"question" : "Which genes or proteins are linked to melanoma and also interact with TNFSF8?", "answer" : "melanoma|TNFSF8"},
        ]
PRIME_INSTRUCTION = "You are a knowledgeable assistant which identifies medical entities in the given sentences. Separate entities using '|'."

MAG_MULTISHOT_EXAMPLES = [
            {"question": "Could you find research articles on the measurement of radioactive gallium isotopes disintegration rates?",
             "answer": "FieldOfStudy:measurement of radioactive gallium isotopes disintegration rates"},
            {"question": "What research on water absorption in different frequency ranges have been referenced or deemed significant in the paper entitled 'High-resolution terahertz atmospheric water vapor continuum measurements'",
             "answer": "FieldOfStudy:water absorption in different frequency ranges\nPaper:High-resolution terahertz atmospheric water vapor continuum measurements"},
            {"question": "Publications by Point Park University authors on stellar populations in tidal tails",
             "answer": "Institution:Point Park University\nFieldOfStudy:stellar populations in tidal tails"},
            {"question": "Show me publications by A.J. Turvey on the topic of supersymmetry particle searches.",
             "answer": "Author:A.J. Turvey\nField of study: supersymmetry particle searches"},
        ]
MAG_INSTRUCTION = "You are a smart assistant which identifies entities in a given questions. There are institutions, authors, fields of study and papers."
MAG_LABELS = ['Institution', 'Author', 'FieldOfStudy', 'Paper']

class NER:
    def __init__(self, driver: Driver, dataset_name: str):
        match dataset_name:
            case 'prime':
                self.dataset_name = dataset_name
                self.multi_shot_examples = PRIME_MULTISHOT_EXAMPLES
                self.system_instruction = PRIME_INSTRUCTION
            case 'mag':
                self.dataset_name = dataset_name
                self.multi_shot_examples = MAG_MULTISHOT_EXAMPLES
                self.system_instruction = MAG_INSTRUCTION
                self.labels = MAG_LABELS
            case _:
                raise ValueError

        self.normal_to_original = defaultdict(list)
        for rec in driver.execute_query("""MATCH (n) RETURN DISTINCT n.name AS name""").records:
            self.normal_to_original[rec['name'].lower()].append(rec['name'])  # map to all original names with specific canonical name

    def find_source_nodes(self, question: str, driver: Driver, openai_api_key: str, model="gpt-4o-mini"):
        match self.dataset_name:
            case 'prime':
                named_entities = self.identify_unlabeled_entities(question, openai_api_key=openai_api_key, model=model)
                matched_nodes = self.match_labeled_entities(driver=driver, entities=named_entities, openai_api_key=openai_api_key)
                return matched_nodes
            case 'mag':
                named_labeled_entities = self.identify_labeled_entities(question, openai_api_key=openai_api_key)
                matched_nodes = self.match_labeled_entities(driver=driver, entities=named_labeled_entities, openai_api_key=openai_api_key)
                return matched_nodes
            case _:
                ...


    def ask_ai(self, question: str, openai_api_key: str, model="gpt-4o-mini"):
        user_model_correspondence = [({"role": "user", "content": f"Q:\"{multi_shot_example['question']}\""},
                                      {"role": "assistant", "content": f"A:{multi_shot_example['answer']}"}) for
                                     multi_shot_example in self.multi_shot_examples]
        user_model_correspondence = [x for xs in user_model_correspondence for x in xs]  # flatten
        if model=="gpt-4o-mini":
            client = OpenAI(api_key=openai_api_key)
            completion = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": self.system_instruction},
                    *user_model_correspondence,
                    {"role": "user", "content": f"Q:\"{question}"},
                ]
            )
            response = completion.choices[0].message.content.lstrip('A').lstrip(':')
        if model=="llama-3.1-8b-instruct":
            pipeline = transformers.pipeline(
                "text-generation",
                model="meta-llama/Meta-Llama-3.1-8B-Instruct",
                model_kwargs={"torch_dtype": torch.bfloat16},
                device_map="auto",
            )
            messages = [
                {"role": "system", "content": self.system_instruction},
                *user_model_correspondence,
                {"role": "user", "content": f"Q:\"{question}"},
            ]
            outputs = pipeline(
                messages,
                max_new_tokens=256,
            )
            response = outputs[0]["generated_text"][-1]

        return response


    def identify_unlabeled_entities(self, question: str, openai_api_key: str, model="gpt-4o-mini") -> list[tuple[str, str]]:
        response = self.ask_ai(question, openai_api_key=openai_api_key, model=model)
        entities = response.lstrip('A').lstrip(':').split('|')
        labeled_entities = [('nameEmbedding', entity) for entity in entities]
        return labeled_entities


    def identify_labeled_entities(self, question: str, openai_api_key: str) -> list[tuple[str, str]]:
        response = self.ask_ai(question, openai_api_key=openai_api_key)

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
            label = response[idx0s[i]:idx1s[i] - 1]
            value = response[idx1s[i]:idx0s[i + 1]] if i < len(idx0s) - 1 else response[idx1s[i]:]
            labeled_entities.append((label, value.strip()))
        return labeled_entities


    def match_labeled_entities(self, driver: Driver, openai_api_key: str, entities: list[tuple[str, str]], k=100):
        matched_entity_names = set()
        for label, string in entities:
            if string.lower() in self.normal_to_original.keys():  # Check if the exact string exists in db, ignoring label
                matched_entity_names |= set(self.normal_to_original[string.lower()])
            else:
                try:  # Otherwise search for the most similar among the nodes of the specified label.
                    res = driver.execute_query("""WITH genai.vector.encode($string, 'OpenAI', { token: $api_key }) AS embedding
                                                  CALL db.index.vector.queryNodes($vectorIndex, $k, embedding) YIELD node
                                                  RETURN node.name AS name""",
                                               parameters_={'string': string, 'k': k, 'api_key': openai_api_key,
                                                            'vectorIndex': label})
                    matched_entity_names.add(res.records[0]['name'])
                except Exception as e:
                    print(e, string)
        return list(matched_entity_names)