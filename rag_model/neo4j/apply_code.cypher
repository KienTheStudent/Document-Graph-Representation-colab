MATCH (ch:Chapter:Test_embedding)
WHERE ch.id ENDS WITH "_no_chapter"
DETACH DELETE ch

MATCH (n:Test_embedding)
where size(keys(n)) < 3
DETACH DELETE n

//Preprocess before projection to ensure all attributes are present
MATCH (n:Test_embedding)
WHERE n.document_type is null
SET n.document_type = 0;

MATCH (n:Test_embedding)
WHERE n.amend is null
SET n.amend = -1;

MATCH (n:Test_embedding)
WHERE n.original_embedding is null
SET n.original_embedding = apoc.convert.toList(reduce(x=[], i IN range(1,768) | x + 0.0));

//Project
CALL gds.graph.project(
  'sample',
  ['*'],
  '*',
  {
    nodeProperties: ['original_embedding', 'amend', 'document_type']
  }
);

//GraphSAGE
CALL gds.beta.graphSage.train(
  'sample',
  {
    modelName: 'sage',
    featureProperties: ['original_embedding', 'document_type', 'amend'],
    // labelProperty: '*',       // node attribute with class labels (target label)
    nodeLabels: ['*'],           // label of nodes used, for example only Clause node
    embeddingDimension: 768,
    aggregator: 'mean',
    epochs: 10,
    learningRate: 0.001,

    projectedFeatureDimension: 768 //for multiple labels per node, we concatenate and project all attributes from all node labels, for example
                                    // label A (a,b,c), label B (d, e) -> projectedFeatureDimension is the maximum attribute of a Node label = 3
  }
) YIELD modelInfo as info
RETURN
  info.modelName as modelName,
  info.metrics.didConverge as didConverge,
  info.metrics.ranEpochs as ranEpochs,
  info.metrics.epochLosses as epochLosses

//Write back
CALL gds.beta.graphSage.write(
  'sample', //projected graph into GDS
  {
    modelName: 'sage', //trained GraphSage model
    writeProperty: 'embedding' //new node attribute saved for Embedding
  }
);
