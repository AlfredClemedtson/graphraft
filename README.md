# GraphRAFT

Codebase for [GraphRAFT: Retrieval Augmented Fine-Tuning for Knowledge Graphs on Graph Databases](https://arxiv.org/abs/2504.05478)

## Setup

[Installing Neo4j](https://neo4j.com/docs/operations-manual/current/installation/)

[Loading a database dump - Neo4j](https://neo4j.com/docs/operations-manual/current/backup-restore/restore-dump/)\
`https://gds-public-dataset.s3.us-east-1.amazonaws.com/prime.dump` \
`https://gds-public-dataset.s3.us-east-1.amazonaws.com/mag.dump`


## Inference
1. Start the Neo4j database.
2. Run `python graphraft.py` with  `--dataset prime` or `--dataset mag`


## Reproduce training and evaluation
1. Preprocess NER and Cypher queries in `training/generate_training_data.ipynb`
2. Train LLM1: `python llm1.py --dataset [] --train`
3. Preprocess cypher generation: `python llm1.py --dataset [] --generate_valid --generate_test`
4. Preprocess retrieval in `training/generate_training_data.ipynb`
5. Train LLM2: `python llm1.py --dataset [] --train`
6. Evaluate pipeline:  `python llm2.py --dataset [] --adapter_dir [] --test`

## Reference
To cite our work, please use
```
@misc{graphraft,
      title={GraphRAFT: Retrieval Augmented Fine-Tuning for Knowledge Graphs on Graph Databases}, 
      author={Alfred Clemedtson and Borun Shi},
      year={2025},
      eprint={2504.05478},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2504.05478}, 
}
```
