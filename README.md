## Setup
[Installing Neo4j](https://neo4j.com/docs/operations-manual/current/installation/)


## Inference
1. Start the Neo4j database.
2. Run `python graphraft.py` with  `--dataset prime` or `--dataset mag`


## Reproduce training and evaluation
1. Preprocess NER and Cypher queries in `training/generate_training_data.ipynb`
2. Train LLM1: `python llm1.py --dataset [] --train`
3. Preprocess cypher generation: `python llm1.py --dataset [] --generate_valid --generate_test`. 
4. Optionally evaluate 3. using `python eval_llm.py --dataset --max_nodes 20`
5. Preprocess retrieval in `training/generate_training_data.ipynb`
6. Train LLM2: `python llm2.py --dataset [] --train`
7. Evaluate pipeline:  `python llm2.py --dataset [] --adapter_dir [] --test --add_more_answers`