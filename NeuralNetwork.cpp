// includes
#include "NeuralNetwork.hpp"
#include "Trace.hpp"
using namespace std;



// NeuralNetwork -----------------------------------------------------------------------------------------------------------------------------------

// STUDENT TODO: IMPLEMENT
void NeuralNetwork::eval() {
    evaluating = true;
}

// STUDENT TODO: IMPLEMENT
void NeuralNetwork::train() {
    evaluating = false;
}

// STUDENT TODO: IMPLEMENT
void NeuralNetwork::setLearningRate(double lr) {
    learningRate = lr;
}

// STUDENT TODO: IMPLEMENT
void NeuralNetwork::setInputNodeIds(std::vector<int> newInputNodeIds) {
    inputNodeIds = newInputNodeIds;
}

// STUDENT TODO: IMPLEMENT
void NeuralNetwork::setOutputNodeIds(std::vector<int> newOutputNodeIds) {

    for ( int id : newOutputNodeIds){
        nodes.at(id)->postActivationValue = 0.5; //i hate whoever decided this was a thing
    }

    outputNodeIds = newOutputNodeIds;
}

// STUDENT TODO: IMPLEMENT
vector<int> NeuralNetwork::getInputNodeIds() const {
    return inputNodeIds;
}

// STUDENT TODO: IMPLEMENT
vector<int> NeuralNetwork::getOutputNodeIds() const {
    return outputNodeIds;
}

// STUDENT TODO: IMPLEMENT
vector<double> NeuralNetwork::predict(DataInstance instance) {

    vector<double> input = instance.x;

    // error checking : size mismatch

    //cout << "a" << endl;
    if (input.size() != inputNodeIds.size()) {
        cerr << "input size mismatch." << endl;
        cerr << "\tNeuralNet expected input size: " << inputNodeIds.size() << endl;
        cerr << "\tBut got: " << input.size() << endl;
        return vector<double>();
    }

    //layers
    //yes so layers[0] = inputNodeIds i think

    //cout << "b" << endl;

    int layerc = layers.size();
    queue<int> bfsq;
    //queue<int> bfsq2;

    //cout << layerc << endl;
    
    for (int i = 0; i < inputNodeIds.size(); i++){
        //cout << inputNodeIds[i] << " " << input[i] << endl;

        int id = inputNodeIds[i];
        double value = input[id];

        NodeInfo* curr = (nodes.at(id));

        curr->postActivationValue = value;
        bfsq.push(id);
    }
    
    /*
    for (int id : inputNodeIds) {
        //process input value
        NodeInfo* curr = (nodes.at(id));

        //we set post activation directly bc its the starting thing and it also has no bias
        curr->postActivationValue = input[id];

        bfsq.push(id);
    }
    */

    //so we set up our initial queue, now it's time for a while loop
    //we will go down 1 level and stuff it all in a queue. 
    //Before going to the next level, let's consolidate the values we iterated through
    //cout << "c" << endl;

    for (int i = 0; i < layerc-1; i++){
        vector<int> currentLayer = layers[i];
        vector<int> nextLayer = layers[i+1];

        //cout << "d" << endl;

        while (!bfsq.empty()){
            int startNodeIndex = bfsq.front();

            NodeInfo* startNode = (nodes.at(startNodeIndex));
            double startNodeRawVal = startNode->postActivationValue;

            //for every node in the next layer, add the value from the current node to it times the weighted value
            for (int j = 0; j < nextLayer.size(); j++){
                int endNodeIndex = nextLayer[j];

                //now get the edge with the two known indicies
                Connection edge = adjacencyList[startNodeIndex][endNodeIndex];

                NodeInfo* endNode = (nodes.at(endNodeIndex));

                endNode->preActivationValue += startNodeRawVal * edge.weight;

            }

            bfsq.pop();
        }
        
        //create the new queue for the next iteration
        //also activate all these nodes
        //cout << "e" << endl;

        for (int j = 0; j < nextLayer.size(); j++){
            int endNodeIndex = nextLayer[j];
            NodeInfo* curr = (nodes.at(endNodeIndex));
            curr->preActivationValue += curr->bias;
            curr->activate();
            
            bfsq.push(endNodeIndex);
        }
        
        //cout << "f" << endl;
    }

    //now after this, bfsq contains the output nodes only with proper postActivationValues


    // BFT implementation goes here.
    // Note: before traversal begins, each input value in `input` must be loaded into
    // the corresponding input node's postActivationValue. Input nodes are not activated —
    // their value is passed forward directly.
    // Use visitPredictNode and visitPredictNeighbor to handle the neural network math
    // at each step of your traversal.

    //so for each value in 'input' there is a corresponding input node...


    vector<double> output;
    for (int i = 0; i < outputNodeIds.size(); i++) {
        int dest = outputNodeIds.at(i);
        NodeInfo* outputNode = nodes.at(dest);
        output.push_back(outputNode->postActivationValue);
    }

    if (evaluating) {
        flush();
    } else {
        // increment batch size
        batchSize++;
        // accumulate derivatives. If in training mode, weights and biases get accumulated
        contribute(instance.y, output.at(0));
    }
    return output;
}

//Root function of the self-evaluation chain during model training.
bool NeuralNetwork::contribute(double y, double p) {

    //for each input node, start a DFT to eventually backtrack and acumulates connection weight and node bias deltas for later application.
    for (int id : layers[0]) { contribute(id,y,p); }

    //By now we have completed all DFTs through the Neural Network.
    //This function empties the vector of saved contributeOuts
    //It also resets the current values of all nodes. (preactivationval == postactivationval == 0)
    flush(); 

    return true;
}

//Recursive helper function for the previous function of the same name.
double NeuralNetwork::contribute(int nodeId, const double& y, const double& p) {
    // don't remove this line, used for visualization
    visitContributeStart(nodeId); 
    
    //declaration of important values
    double incomingContribution = 0;
    double outgoingContribution = 0;

    //iterate through each node this self has an outgoing connection to.
    //if there are none, skip straight to backpropagation.
    for (pair<const int, Connection>& edge : adjacencyList[nodeId]){
        //node here refers a node that self has an outgoing connection to.

        //if node was already processed in another recursive branch, 
        //copy the precomputed contribution instead of running that again.
        if (contributions.find(edge.second.dest) != contributions.end()){
            incomingContribution = contributions[edge.second.dest];
        //Otherwise, recurse to get the contribution from node
        } else {
            incomingContribution = contribute(edge.second.dest,y,p);
        }
        //Once we have this contribution, we can compute the delta of the connection between node and self using this function
        visitContributeNeighbor(edge.second, incomingContribution, outgoingContribution);
    }

    //After recieving some sort of incoming contribution, or if we have no neighbors (self is output node), we need to calculate our outgoing contribution
    //check if self is an output node, outgoing contribution is calculated directly from this formula
    if (adjacencyList.at(nodeId).empty()) {
        outgoingContribution = -1 * ((y - p) / (p * (1 - p)));
    }
    //For all nodes, process outgoing contribution using the provided method.
    visitContributeNode(nodeId, outgoingContribution);
    //Save this outgoing contribution, so we dont increment it twice and there's less overall recursive processing.
    contributions[nodeId] = outgoingContribution;

    //return the outgoing contribution, it will become the incomingContribution of another node.
    return outgoingContribution;
}

//applies the deltas of all nodes and connections onto their respective bias and weight values.
bool NeuralNetwork::update() {
    //BFT, same one used in prediction.
    //First, create a queue for travering breadth and put all the input nodes inside it.
    int layerc = layers.size();
    queue<int> bfsq;
    for (int id : layers[0]) {
        bfsq.push(id);
    }
    //iterate through each later (breadth)
    for (int i = 0; i < layerc; i++){
        while (!bfsq.empty()){
            //get the node at the front of the queue
            int nid = bfsq.front();
            NodeInfo* n = (nodes.at(nid));

            //apply delta of this node onto its bias, the formula is bias = bias - (learningRate * delta)
            //but dont modify the bias of input nodes
            if(i != 0){
                n -> bias -= n -> delta * learningRate;
                n -> delta = 0;
            }
            //for every outgoing connection of this node, apply delta of this connection onto its weight. The formula is weight = weight - (learningRate * delta)
            for (pair<const int, Connection>& c : (adjacencyList[nid])){
                c.second.weight -= c.second.delta * learningRate;
                c.second.delta = 0;
            }
            //pop this node off the queue, so we can process the next node in the layer
            bfsq.pop();
        }
        
        //if there are more layers, push all the layer's nodes to the queue
        //this is the next breadth we are traversing
        if (i+1 < layers.size()){
            for (int j = 0; j < layers[i+1].size(); j++){
                int endNodeIndex = layers[i+1][j];
                bfsq.push(endNodeIndex);
            }
        }
    }
    //Once again, this function empties the vector of saved contributeOuts
    //It also resets the current values of all nodes. (preactivationval == postactivationval == 0)
    flush();
    //return success
    return true;
}




// Feel free to explore the remaining code, but no need to implement past this point

// ----------- YOU DO NOT NEED TO TOUCH THE REMAINING CODE -----------------------------------------------------------------
// ----------- YOU DO NOT NEED TO TOUCH THE REMAINING CODE -----------------------------------------------------------------
// ----------- YOU DO NOT NEED TO TOUCH THE REMAINING CODE -----------------------------------------------------------------
// ----------- YOU DO NOT NEED TO TOUCH THE REMAINING CODE -----------------------------------------------------------------
// ----------- YOU DO NOT NEED TO TOUCH THE REMAINING CODE -----------------------------------------------------------------







// Constructors
NeuralNetwork::NeuralNetwork() : Graph(0) {
    learningRate = 0.1;
    evaluating = false;
    batchSize = 0;
}

NeuralNetwork::NeuralNetwork(int size) : Graph(size) {
    learningRate = 0.1;
    evaluating = false;
    batchSize = 0;
}

NeuralNetwork::NeuralNetwork(string filename) : Graph() {
    // open file
    ifstream fin(filename);

    // error check
    if (fin.fail()) {
        cerr << "Could not open " << filename << " for reading. " << endl;
        exit(1);
    }

    // load network
    loadNetwork(fin);
    learningRate = 0.1;
    evaluating = false;
    batchSize = 0;

    // close file
    fin.close();
}

NeuralNetwork::NeuralNetwork(istream& in) : Graph() {
    loadNetwork(in);
    learningRate = 0.1;
    evaluating = false;
    batchSize = 0;
}

const vector<vector<int> >& NeuralNetwork::getLayers() const {
    return layers;
}

void NeuralNetwork::loadNetwork(istream& in) {
    int numLayers(0), totalNodes(0), numNodes(0), weightModifications(0), biasModifications(0); string activationMethod = "identity";
    string junk;
    in >> numLayers; in >> totalNodes; getline(in, junk);
    if (numLayers <= 1) {
        cerr << "Neural Network must have at least 2 layers, but got " << numLayers << " layers" << endl;
        exit(1);
    }

    // resize network to accomodate expected nodes.
    resize(totalNodes);
    this->size = totalNodes;

    int currentNodeId(0);

    vector<int> previousLayer;
    vector<int> currentLayer;
    for (int i = 0; i < numLayers; i++) {
        currentLayer.clear();
        //  For each layer

        // get nodes for this layer and activation method
        in >> numNodes; in >> activationMethod; getline(in, junk);

        for (int j = 0; j < numNodes; j++) {
            // For every node, add a new node to the network with proper activationMethod
            // initialize bias to 0.
            updateNode(currentNodeId, NodeInfo(activationMethod, 0, 0));
            // This node has an id of currentNodeId
            currentLayer.push_back(currentNodeId++);
        }

        if (i != 0) {
            // There exists a previous layer, now we set out connections
            for (int k = 0; k < previousLayer.size(); k++) {
                for (int w = 0; w < currentLayer.size(); w++) {

                    // Initialize an initial weight of a sample from the standard normal distribution
                    updateConnection(previousLayer.at(k), currentLayer.at(w), sample());
                }
            }
        }

        // Crawl forward.
        previousLayer = currentLayer;
        layers.push_back(currentLayer);
    }
    in >> weightModifications; getline(in, junk);
    int v(0),u(0); double w(0), b(0);

    // load weights by updating connections
    for (int i = 0; i < weightModifications; i++) {
        in >> v; in >> u; in >> w; getline(in , junk);
        updateConnection(v, u, w);
    }

    in >> biasModifications; getline(in , junk);

    // load biases by updating node info
    for (int i = 0; i < biasModifications; i++) {
        in >> v; in >> b; getline(in, junk);
        NodeInfo* thisNode = getNode(v);
        thisNode->bias = b;
    }

    setInputNodeIds(layers.at(0));
    setOutputNodeIds(layers.at(layers.size()-1));
}

// visitPredictNode: called when your BFT dequeues a node.
// It completes the computation for this node by:
//   1. Adding the bias to the accumulated weighted sum (preActivationValue)
//   2. Applying the activation function and storing the result in postActivationValue
// After this call, the node's output value (postActivationValue) is ready to be
// passed forward to the next layer via visitPredictNeighbor.
void NeuralNetwork::visitPredictNode(int vId) {
    // accumulate bias, and activate
    NodeInfo* v = nodes.at(vId);
    v->preActivationValue += v->bias;
    v->activate();
    // visualization use
    if (viz::isTracing()) {
        viz::traceNodeState(0, "forward", vId,
                            v->preActivationValue,
                            v->postActivationValue,
                            v->bias,
                            v->delta,
                            "current");
    }
}

// visitPredictNeighbor: called for each outgoing connection from a dequeued node.
// It accumulates one term of the weighted sum into the destination node:
//   dest.preActivationValue += source.postActivationValue * weight
// This must be called for ALL incoming connections to a node before
// visitPredictNode is called on that node — which is why BFT is required:
// it ensures a whole layer's outputs are ready before the next layer is processed.
void NeuralNetwork::visitPredictNeighbor(Connection c) {
    NodeInfo* v = nodes.at(c.source);
    NodeInfo* u = nodes.at(c.dest);
    double w = c.weight;
    u->preActivationValue += v->postActivationValue * w;
    // visualization use
    if (viz::isTracing()) {
        viz::traceEdgeState(0, "forward",
                            c.source,
                            c.dest,
                            c.weight,
                            c.delta);
        viz::traceNodeState(0, "forward", c.dest,
                            u->preActivationValue,
                            u->postActivationValue,
                            u->bias,
                            u->delta,
                            "neighbor");
    }
}

// visitContributeStart: called at the start of the contribution step for a node.
void NeuralNetwork::visitContributeStart(int vId) {
    NodeInfo* v = nodes.at(vId);
    // visualization use
    if (viz::isTracing()) {
        viz::traceNodeState(0, "backward", vId,
                            v->preActivationValue,
                            v->postActivationValue,
                            v->bias,
                            v->delta,
                            "stack");
    }
}
// visitContributeNode: called after all neighbors of a node have been visited during DFT.
// outgoingContribution at this point holds the sum of weighted incoming contributions
// from the next layer. This function:
//   1. Multiplies outgoingContribution by the activation derivative at this node
//      (chain rule: how much did this node's activation affect the error?)
//   2. Accumulates that result into the node's delta (gradient for its bias)
// After this call, outgoingContribution holds the value to be passed back to
// the previous layer as their incomingContribution.
void NeuralNetwork::visitContributeNode(int vId, double& outgoingContribution) {
    NodeInfo* v = nodes.at(vId);
    outgoingContribution *= v->derive();

    //contribute bias derivative
    v->delta += outgoingContribution;
    // visualization use
    if (viz::isTracing()) {
        viz::traceNodeState(0, "backward", vId,
                            v->preActivationValue,
                            v->postActivationValue,
                            v->bias,
                            v->delta,
                            "current");
    }
}

// visitContributeNeighbor: called for each outgoing connection during DFT, before visitContributeNode.
// incomingContribution is the contribution returned by the recursive call on the neighbor (next layer).
// This function:
//   1. Adds weight * incomingContribution to outgoingContribution
//      (this node's share of the error flowing back from the neighbor)
//   2. Accumulates the weight gradient into c.delta
//      (how much should this weight change? proportional to incomingContribution * this node's output)
void NeuralNetwork::visitContributeNeighbor(Connection& c, double& incomingContribution, double& outgoingContribution) {
    NodeInfo* v = nodes.at(c.source);
    // update outgoingContribution
    outgoingContribution += c.weight * incomingContribution;

    // accumulate weight derivative
    c.delta += incomingContribution * v->postActivationValue;
    // visualization use
    if (viz::isTracing()) {
        viz::traceEdgeState(0, "backward",
                            c.source,
                            c.dest,
                            c.weight,
                            c.delta);
        viz::traceNodeState(0, "backward", c.source,
                            v->preActivationValue,
                            v->postActivationValue,
                            v->bias,
                            v->delta,
                            "neighbor");
    }
}

void NeuralNetwork::flush() {
    // set every node value to 0 to refresh computation.
    for (int i = 0; i < nodes.size(); i++) {
        nodes.at(i)->postActivationValue = 0;
        nodes.at(i)->preActivationValue = 0;
    }
    contributions.clear();
    batchSize = 0;
}

double NeuralNetwork::assess(string filename) {
    DataLoader dl(filename);
    return assess(dl);
}

double NeuralNetwork::assess(DataLoader dl) {
    bool stateBefore = evaluating;
    evaluating = true;
    double count(0);
    double correct(0);
    vector<double> output;
    for (int i = 0; i < dl.getData().size(); i++) {
        DataInstance di = dl.getData().at(i);
        output = predict(di);
        if (static_cast<int>(round(output.at(0))) == di.y) {
            correct++;
        }
        count++;
    }

    if (dl.getData().empty()) {
        cerr << "Cannot assess accuracy on an empty dataset" << endl;
        exit(1);
    }
    evaluating = stateBefore;
    return correct / count;
}


void NeuralNetwork::saveModel(string filename) {
    ofstream fout(filename);
    
    fout << layers.size() << " " << getNodes().size() << endl;
    for (int i = 0; i < layers.size(); i++) {
        NodeInfo* layerNode = getNodes().at(layers.at(i).at(0));
        string activationType = getActivationIdentifier(layerNode->activationFunction);

        fout << layers.at(i).size() << " " << activationType << endl;
    }

    int numWeights = 0;
    int numBias = 0;
    stringstream weightStream;
    stringstream biasStream;
    for (int i = 0; i < nodes.size(); i++) {
        numBias++;
        biasStream << i << " " << nodes.at(i)->bias << endl;

        for (auto j = adjacencyList.at(i).begin(); j != adjacencyList.at(i).end(); j++) {
            numWeights++;
            weightStream << j->second.source << " " << j->second.dest << " " << j->second.weight << endl;
        }
    }

    fout << numWeights << endl;
    fout << weightStream.str();
    fout << numBias << endl;
    fout << biasStream.str();

    fout.close();


}

ostream& operator<<(ostream& out, const NeuralNetwork& nn) {
    for (int i = 0; i < nn.layers.size(); i++) {
        out << "layer " << i << ": ";
        for (int j = 0; j < nn.layers.at(i).size(); j++) {
            out << nn.layers.at(i).at(j) << " ";
        }
        out << endl;
    }
    // outputs the nn in dot format
    out << static_cast<const Graph&>(nn) << endl;
    return out;
}

