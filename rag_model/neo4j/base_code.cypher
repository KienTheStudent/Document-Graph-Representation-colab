// MODE in Neo4j GDS
// Stream Mode: return result directly to the client (Cypher result stream) but doesn't modify the in-memory graph or Database
// -> only use when inspect result, test an algorithm or pipe results into structure

// Write Mode: Write the algorithm's results into Neo4j Database as node properties/ relationship properties
// -> direct write to disk so will be slower

// Mutate Mode: Modify the projected in-memory GDS graph, not the Neo4j Database
// -> add properties into the GDS projection, faster than write but written properties will disappear when in-memory graph is dropped


// Graph preparation
CALL gds.graph.project(
  'legalGraph', // name of the graph
  'Clause', //Node label, for example only Clause nodes
  'CITES', //Types of relations to be included in the projection
  {
    nodeProperties: [
        'feature0',
        'feature1',
        'feature2'
    ]
  }
);

//List Graph properties
CALL gds.graph.list() 
YIELD graphName, nodeCount, relationshipCount
RETURN graphName, nodeCount, relationshipCount
ORDER BY graphName ASC

CALL gds.graph.list('personsCypher')
YIELD graphName, configuration, schemaWithOrientation
RETURN graphName, configuration.query AS query, schemaWithOrientation

// New Graph projection in recent version
MATCH (source:Person)-[r:LIKES]->(target:Instrument)
RETURN gds.graph.project(
  'persons_with_instruments',
  source,
  target,
  {
    sourceNodeLabels: labels(source),
    sourceNodeProperties: source { .age, .heightAndWeight }, // use . before node attribute to take the actual value without the need for aliasing
    targetNodeLabels: labels(target),
    targetNodeProperties: target { .cost },
    relationshipType: type(r),
    relationshipProperties: r { .relWeight }
  },
  { undirectedRelationshipTypes: ['LIKES'] }
)

// Drop projected graphs
CALL gds.graph.drop('yourGraphName')
YIELD graphName, nodeCount, relationshipCount
RETURN graphName, nodeCount, relationshipCount

//
// 

//List models 
CALL gds.model.list();

// or List model by types
CALL gds.beta.graphSage.list(); //only GraphSage
CALL gds.beta.pipeline.nodeClassification.list(); //only node Classification
CALL gds.beta.pipeline.nodeEmbedding.list(); //only nodeEmbedding
CALL gds.beta.pipeline.linkPrediction.list(); //only linkPrediction

// drop a specific model
CALL gds.model.drop('yourModelName');

// or drop all model
CALL gds.model.list()
YIELD modelName
CALL gds.model.drop(modelName) YIELD modelInfo
RETURN modelName, modelInfo;

//
//
// GraphSAGE
CALL gds.beta.graphSage.train(
  'persons', //persons is the projected graph into GDS
  {
    modelName: 'sage_model', // name of the model
    featureProperties: ['age', 'heightAndWeight'], // node feature to calculate embedding, must be numerical
    aggregator: 'pool', // mean/ pool
    activationFunction: 'sigmoid',
    embeddingDimension: 128, //256, 512
    randomSeed: 42,
    sampleSizes: [25, 10],
    epochs: 10,
    batchSize: 1,
    learningRate: 0.01,
    negativeSamplingRatio: 5, //for unsupervised (embedding training), for each real edge,
                                    // generate 5 fake edges for model enhancement
    projectedFeatureDimension: 3 //for multiple labels per node, we concatenate and project all attributes from all node labels, for example
                                    // label A (a,b,c), label B (d, e) -> projectedFeatureDimension is the maximum attribute of a Node label = 3
  }
) YIELD modelInfo as info
RETURN
  info.modelName as modelName,
  info.metrics.didConverge as didConverge,
  info.metrics.ranEpochs as ranEpochs,
  info.metrics.epochLosses as epochLosses

// show information of the Model training
CALL gds.model.list('model')
yield modelInfo as info
return info

// write graphsage embedding to neo4j, save into neo4j
CALL gds.beta.graphSage.write(
  'myGraph', //projected graph into GDS
  {
    modelName: 'sageModel', //trained GraphSage model
    writeProperty: 'Embedding' //new node attribute saved for Embedding
  }
);

// -> can use KNN, cosine sim, vector search or subgraph retrieval using embedding similarity

// Optimal choice to encode/ embed Legal text first using pretrained strong embedding model first
// save that as node.original_embedding then use GraphSage on top of that text embedding
// to enhance the embedding

// project a graph using the semantic embedding and filter for only certain types of nodes
CALL gds.graph.project(
  'legalGraph', // name of the graph
  'Clause', //Node label, for example only Clause nodes
  'CITES', //Types of relations to be included in the projection
  {
    nodeProperties: [
        'original_embedding', //use the node attribute embedding
        'feature1',
        'feature2'
    ]
  }
);

CALL gds.beta.graphSage.train(
  'legalGraph',
  {
    modelName: 'comprehensive_embedding',
    featureProperties: ['original_embedding', 'feature1', 'feature2'],
    embeddingDimension: 256,
    epochs: 20
  }
);

//
// Use GraphSage for Supervised model
CALL gds.beta.graphSage.train(
  'myGraph',
  {
    modelName: 'sageClassifierModel',
    featureProperties: ['textEmbedding', 'clause_length', 'num_citations'],
    labelProperty: 'labelProp',       // node attribute with class labels (target label)
    nodeLabels: ['Clause'],           // label of nodes used, for example only Clause node
    embeddingDimension: 128,
    aggregator: 'mean',
    epochs: 20,
    learningRate: 0.001,
    validationFolds: 5
  }
);

CALL gds.beta.graphSage.predict.stream(
  'myGraph',                            //projected graph into GDS
  { modelName: 'sageClassifierModel' }
)
YIELD nodeId, predictedLabel, probability
RETURN gds.util.asNode(nodeId).id AS node,
       predictedLabel, probability;

//
// For Graph with Relation Weight
CALL gds.beta.graphSage.train(
  'persons',
  {
    modelName: 'weightedTrainedModel',
    featureProperties: ['age', 'heightAndWeight'],
    relationshipWeightProperty: 'relWeight', //relation Weight, assigned at the creation of the Node and node connection edges
    nodeLabels: ['Person'],
    relationshipTypes: ['KNOWS']
  }
)

//
// Working with FastRP

// Projection first
MATCH (source:Person)-[r:BUYS]->(target:Product)
RETURN gds.graph.project(
  'purchases',
  source,
  target,
  {
    sourceNodeLabels: labels(source),
    targetNodeLabels: labels(target),
    relationshipType: 'BUYS',
    relationshipProperties: r { .amount }
  },
  { undirectedRelationshipTypes: ['BUYS'] } //Note that the Relationship BUYS has been put into Undirected Relationship, as a default choice when using FastRP
)

// then write the embedding as mutate mode to only change the in-memory graph
CALL gds.fastRP.mutate(
  'purchases',
  {
    embeddingDimension: 4,
    iterationWeights: [0.8, 1, 1, 1],
    relationshipWeightProperty: 'amount',
    randomSeed: 42,
    mutateProperty: 'embedding'
  }
)
YIELD nodePropertiesWritten

// Run KNN and return the score back to the graph
CALL gds.knn.write(
  'purchases',
  {
    nodeProperties: ['embedding'],
    nodeLabels: ['Person'],
    topK: 2,
    sampleRate: 1.0,
    deltaThreshold: 0.0,
    randomSeed: 42,
    concurrency: 1,
    writeProperty: 'score',
    writeRelationshipType: 'SIMILAR'
  }
)
YIELD similarityDistribution
RETURN similarityDistribution.mean AS meanSimilarity

//
// Working with Node2Vec
// Node2Vec is a node embedding algorithm that computes a vector representation of a node based on Random Walk in the graph. The neighborhood is sampled through random 
// walks. Using a number of random neighborhood samples, the algorithm trains a single hidden-layer NN to predict the likelihood that a node will occur in a walk based
// on occurences of another node


//
//
// Link Prediction ML pipeline

// Create pipeline and add into the Pipeline Catalog
CALL gds.beta.pipeline.linkPrediction.create('pipe')

// add link features
CALL gds.beta.pipeline.linkPrediction.addFeature(
  'pipe',
  'l2', // feature type l2 for l2-norm
  { nodeProperties: ['age'] }
)

//train-test-split and number of folds for CV
CALL gds.beta.pipeline.linkPrediction.configureSplit(
  'pipe',
  {
    testFraction: 0.25,
    trainFraction: 0.6,
    validationFolds: 3
  }
)

// add model candidate

// Classification 
  // beta
    // LogisticRegression
    // RandomForest
  // alpha
    // MLP
// Regression
  // alpha
    // RandomForest
    // Linear Regression
CALL gds.beta.pipeline.linkPrediction.addLogisticRegression('pipe') //this is LogisticRegression with no further configurations
CALL gds.beta.pipeline.linkPrediction.addRandomForest('pipe') // for RandomForest
CALL gds.beta.pipeline.linkPrediction.addMLP('pipe') //for multilayer perceptron

//train the Model
CALL gds.beta.pipeline.linkPrediction.train(
  'friends',
  {
    pipeline: 'pipe',
    modelName: 'lp-pipeline-model', // manually set the name
    targetRelationshipType: 'KNOWS',
    metrics: ['AUCPR'],
    randomSeed: 42
  }
)
YIELD modelInfo
RETURN
  modelInfo.bestParameters AS winningModel,
  modelInfo.metrics.AUCPR.train.avg AS avgTrainScore,
  modelInfo.metrics.AUCPR.validation.avg AS avgValidationScore,
  modelInfo.metrics.AUCPR.outerTrain AS outerTrainScore,
  modelInfo.metrics.AUCPR.test AS testScore

// Use the model for prediction
CALL gds.beta.pipeline.linkPrediction.predict.stream(
  'friends', // the projected graph
  {
    modelName: 'lp-pipeline-model', // name of the model set above
    topN: 5
  }
)
YIELD node1, node2, probability
RETURN
  gds.util.asNode(node1).name AS person1,
  gds.util.asNode(node2).name AS person2,
  probability
ORDER BY probability DESC, person1