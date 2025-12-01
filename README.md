# Document-Graph-Representation

```bash
git clone https://github.com/GinHikat/Document-Graph-Representation.git

cd Document-Graph-Representation

pip install -r requirement.txt
```

Note that the requirement.txt may miss several libraries, fill them later if missing dependencies occurs. For further work with Pytorch models, torch-related libraries need to be installed later

Also, the files .env, ggsheet_credentials.json and neo4j_credentials.txt will be provided earlier if asked.

I. About the Github file structure

1. rag_model

1.1. Cypher: Working with Neo4j 
+ Install "Neo4j for VS Code" and "Neo4j Viz" extensions in VScode
+ for_neo4j.ipynb: For sample connection to Neo4j and Query via Python code
+ test_viz.cypher: cypher query to direct query to Neo4j, may require connecting to the AuraDB first

1.2. model
+ Modular Pipeline for Autoprocessing document

+ Test code is in test_pipeline.ipynb

+ Retrieval codebase in retrieval_pipeline

2. shared_functions

- gg_sheet_drive.py: Working with gg_sheet and gg_drive, read description in the file

- global functions.py: Working with S3, Neo4j, MySQL (unavailable) and file format conversion

- supabase.py: Working with supabase Relationdb, not necessary now

3. source

3.1. data: Training dataset for RE and NER models

3.2. document: Legal document as corpus

II. Working file for each task

- For working with document corpus: Read the rag_mode/test_google_drive.ipynb

- For working with retrieval pipeline: rag_model/retrieval_pipeline

- For the document autoprocessing into Graphdb: rag_model/model/test_pipeline.ipynb

