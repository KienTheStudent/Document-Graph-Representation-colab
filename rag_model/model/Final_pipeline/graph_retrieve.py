import sys, os

project_root = os.path.abspath(os.path.join(os.getcwd(), "../.."))
if project_root not in sys.path:
    sys.path.append(project_root)

from rag_model.model.RE.final_re import *
from neo4j import GraphDatabase, Result

phobert = PhoBertEmbedding()

url = os.getenv('NEO4J_URI')
username = 'neo4j'
password = os.getenv('NEO4J_AUTH')

driver = GraphDatabase.driver(url, auth=(username, password))

def query_neo4j(text, mode = 1, graph = None, chunks = None, hop = 2, namespace = "Test_embedding"):
    '''
    Retrieve list of top k contexts from Graph
    Parameter:
    text:   Input prompt
    mode:   1: "default",
            2: "traverse_embed",
            3: "traverse_exact",
            4: "pagerank_embed",
            5: "pagerank_exact",
            6: "exact_match",
            7: "exact_match_with_rerank"
            
    graph: use Graph Embedding or not, if None then use Node embedding
    chunks: use chunks or not, if None then use small Node
    hop: number of steps level from original nodes in Traversal
    '''
    if chunks is not None:
        additional_label = "Chunk"
    else: 
        additional_label = ""
        
    labels = ":".join(
        [lbl for lbl in [namespace, additional_label] if lbl]
    )
        
    if graph is not None:
        embedding = "embedding"
    else:
        embedding = 'original_embedding'
    
    mode_dict = {
        1: "default",
        2: "traverse_embed",
        3: "traverse_exact",
        4: "pagerank_embed",
        5: "pagerank_exact",
        6: "exact_match",
        7: "exact_match_with_rerank"
    }

    chosen_mode = mode_dict[mode]
    
    query_emb = text_embedding(text, 3, phobert)
    
    if chosen_mode == 'default':
        result = driver.execute_query(
            f"""
                WITH $emb AS queryEmbedding
                MATCH (n:{labels})
                WHERE n.embedding IS NOT NULL
                WITH n, gds.similarity.cosine(n.{embedding}, queryEmbedding) AS score
                RETURN n.id AS id, n.text AS text, score
                ORDER BY score DESC
                LIMIT 10;
            """, # type: ignore
            {"emb": query_emb},
            result_transformer_=Result.to_df
        ) # type: ignore
        
    if chosen_mode == 'traverse_exact':
        result = driver.execute_query(
            f"""
                WITH $query AS input
                WITH split(toLower(input), " ") AS words
                MATCH (n:{labels})
                WHERE n.text IS NOT NULL

                // word match
                WITH n, size([word IN words WHERE toLower(n.text) CONTAINS word]) AS match_count
                WHERE match_count > 0
                ORDER BY match_count DESC
                LIMIT 10

                WITH collect(n) AS seeds

                UNWIND seeds AS s

                MATCH (s)-[*1..{hop}]-(nbr)
                WHERE nbr <> s

                WITH s AS seed,
                    nbr
                ORDER BY seed.id, nbr.id   // stable ordering

                WITH seed, COLLECT(DISTINCT nbr)[0..5] AS top_neighbors

                WITH seed,
                    // concatenated text: seed.text + “ ” + neighbor texts
                    seed.text + " " + apoc.text.join([x IN top_neighbors | x.text], " ") AS combined_text

                RETURN seed.id AS seed_id,
                    combined_text
                LIMIT 20;

            """, # type: ignore
            {"query": text},
            result_transformer_=Result.to_df
        )# type: ignore 
        
    if chosen_mode == 'traverse_embed':
        result = driver.execute_query(
            f"""
                WITH $emb AS queryEmbedding
                MATCH (n:{labels})
                WHERE n.embedding IS NOT NULL
                WITH n, gds.similarity.cosine(n.{embedding}, queryEmbedding) AS score
                RETURN n.id AS id, n.text AS text, score
                ORDER BY score DESC
                LIMIT 10;

                WITH collect(n) AS seeds

                UNWIND seeds AS s

                MATCH (s)-[*1..{hop}]-(nbr)
                WHERE nbr <> s

                WITH s AS seed,
                    nbr
                ORDER BY seed.id, nbr.id   // stable ordering

                WITH seed, COLLECT(DISTINCT nbr)[0..5] AS top_neighbors

                WITH seed,
                    // concatenated text: seed.text + “ ” + neighbor texts
                    seed.text + " " + apoc.text.join([x IN top_neighbors | x.text], " ") AS combined_text

                RETURN seed.id AS seed_id,
                    combined_text
                LIMIT 20;

            """, # type: ignore
            {"emb": query_emb},
            result_transformer_=Result.to_df
        )# type: ignore    
    
    if chosen_mode == 'exact_match':
        result = driver.execute_query(
            f'''
                WITH $query AS input, $emb AS queryEmbedding
                WITH split(toLower(input), " ") AS words, queryEmbedding
                MATCH (n:{labels})
                WHERE n.text IS NOT NULL

                // Count how many words from input appear in n.text
                WITH n, size([word IN words WHERE toLower(n.text) CONTAINS word]) AS match_count, gds.similarity.cosine(n.{embedding}, queryEmbedding) AS score
                WHERE match_count > 0  // optional: only nodes with at least one match

                RETURN n.id AS id, n.text AS text, match_count, score
                ORDER BY match_count DESC
                LIMIT 10;
            ''', # type: ignore
            {"query": text, "emb":query_emb},
            result_transformer_=Result.to_df
        )# type: ignore

    if chosen_mode == 'exact_match_with_rerank':
        result = driver.execute_query(
            f'''
                WITH $query AS input, $emb AS queryEmbedding
                WITH split(toLower(input), " ") AS words, queryEmbedding

                MATCH (n:{labels})
                WHERE n.text IS NOT NULL AND n.embedding IS NOT NULL

                //Count matching words
                WITH n, size([word IN words WHERE toLower(n.text) CONTAINS word]) AS match_count, queryEmbedding
                WHERE match_count > 0

                //Keep top 20 by word match count
                ORDER BY match_count DESC
                LIMIT 20

                //Compute cosine similarity with query embedding
                WITH n, match_count, gds.similarity.cosine(n.{embedding}, queryEmbedding) AS sim_score

                //Rerank by embedding similarity
                RETURN n.id AS id, n.text AS text, match_count, sim_score
                ORDER BY sim_score DESC
                LIMIT 10;
            ''', # type: ignore
            {"query": text, "emb": query_emb},
            result_transformer_=Result.to_df
        )# type: ignore
         
    return result

