#include <iostream> 
#include <vector> 
#include <cstdlib>
#include<cassert>
#include<cmath>
#include <fstream>
#include <sstream>

using namespace std;

/*---------------------- For handling training data file --------------------------*/

class TrainingData
{
public:
    TrainingData(const string fileName);
    void getTrainingData(vector<double> &input, double &target);
    bool isEOF(void) { return file.eof(); }
    void close(void) { file.close(); }
    void startOver(void);
private:
    ifstream file;
    vector<double> row;
};

TrainingData::TrainingData(const string fileName)
{
    file.open(fileName);
    file.ignore(100, '\n'); // Skip the first row
}

void TrainingData::getTrainingData(vector<double> &input, double &target)
{
    input.clear();
    row.clear();

    string line, val;   
    getline(file, line);
    stringstream str(line);
    while (getline(str, val, ',')) {
        row.push_back(stod(val));
    }

    input.push_back(row[0]);
    input.push_back(row[1]);

    target = row[2];
}

void TrainingData::startOver() {
    file.clear();
    file.seekg(0, ios::beg);
    file.ignore(100, '\n'); // Skip the first row
}

/*---------------------- Building the neural newtwork --------------------------*/

struct Connection {
	double weight;
	double dw; // Derivative of loss function w.r.t each weight: dL/dw
	double da_prev; // Derivative of loss function w.r.t input (@ hidden and output layer): dL/da_prev
};

class Neuron;
typedef vector<Neuron> Layer;


/*---------------------- Neuron class--------------------------*/

class Neuron
{
public:
	Neuron(int numOfInputs, int index);
	void setOutput(double val) { output = val; } // Setter
	double getOutput(void) const { return output; } // Getter
	void feedForward(const Layer &prevLayer, string activation);
	void calcOutputGradients(double target, const Layer &prevLayer, string activation);
	void calcHiddenGradients(const Layer &nextLayer, const Layer &prevLayer, string activation);
	void updateWeights(void);

private:
	int neuronIndex;
	static double lr; // Learning rate
	double output; // a = g(z)
	vector<Connection> neuronWeights; // For weights (w) & gradients: dL/dw, dL/da_prev
	double bias;
	double da, dz, db; // Gradients: dL/da, dL/dz, dL/db
	static double transferFunction(double z, string label);
	static double transferFunctionDerivative(double a, string label);
	static double randomWeight(void) { return rand() / double(RAND_MAX); }
};

Neuron::Neuron(int numOfInputs, int index)
{	
	neuronIndex = index;

	for (int c = 0; c < numOfInputs; ++c) {
		neuronWeights.push_back(Connection());
		neuronWeights.back().weight = randomWeight();
	}
	bias = 0.0;
}

double Neuron::lr = 0.1;

double Neuron::transferFunction(double z, string label)
{
	if (label == "tanh") {
	    return tanh(z);
	} else if (label == "sigmoid") {
		double a = 1.0 / (1.0 + exp(-z));
		return a;
	} else {
		cout << "Invalid activation function" << endl;
		exit(1);
	}
}

double Neuron::transferFunctionDerivative(double a, string label)
{
	if (label == "tanh") {
	    return 1.0 - a * a;
	} else if (label == "sigmoid") {
		return a * (1.0 - a);
	} else {
		cout << "Invalid activation function" << endl;
		exit(1);
	}
}

void Neuron::feedForward(const Layer &prevLayer, string activation)
{	
	double sum  = 0.0;
	// z = (w1*x1 + w2*x2 + ...) + bias
	for (int n = 0; n < prevLayer.size(); ++n) {
		sum += neuronWeights[n].weight * prevLayer[n].getOutput();
	}
	sum += bias;

	// a = g(z)
	output = Neuron::transferFunction(sum, activation);
}

void Neuron::calcOutputGradients(double target, const Layer &prevLayer, string activation)
{
	// da = -y/y' +  (1 - y)/(1 - y') // based on logistic loss
	// da = target == 1 ? (- target / output) : ((1 - target) / (1 - output));
	da = target - output; // based on R.M.S. loss

	// dz = da * g'(z)
	dz = da * Neuron::transferFunctionDerivative(output, activation);

	// dw = da * A_prev
	for (int n = 0; n < prevLayer.size(); ++n) {
		neuronWeights[n].dw = dz * prevLayer[n].getOutput();
		// da_prev = dz * w (to calulate hidden gradients needed in previous layer)
		neuronWeights[n].da_prev = dz * neuronWeights[n].weight;
	}

	// db = dz
	db = dz;
}

void Neuron::calcHiddenGradients(const Layer &nextLayer, const Layer &prevLayer, string activation)
{
	// da = SUM(dz from next layer * weights between current layer & next layer)
	da = 0.0;
	for (int n = 0; n < nextLayer.size(); ++n) {
		da += nextLayer[n].neuronWeights[neuronIndex].da_prev;
	}

	dz = da * Neuron::transferFunctionDerivative(output, activation);

	for (int n = 0; n < prevLayer.size(); ++n) {
		neuronWeights[n].dw = dz * prevLayer[n].getOutput();
		neuronWeights[n].da_prev = dz * neuronWeights[n].weight;
	}
	
	db = dz;
}

void Neuron::updateWeights()
{
	for (int n = 0; n < neuronWeights.size(); ++n) {
		neuronWeights[n].weight += lr * neuronWeights[n].dw;
	}

	bias += lr * db;
}

/*---------------------- NeuralNet class --------------------------*/

class NeuralNet
{
public:
	NeuralNet(const vector<int> &shape);
	void forwardProp(const vector<double> &input, double &output);
	void backProp(const double &target);
	double getLoss(const double &output, const double &target);

private:
	vector<Layer> nn_layers; // nn_layers[numOfLayers][numOfNeuronsPerLayer]
};

NeuralNet::NeuralNet(const vector<int> &shape)
{
	int numOfLayers = shape.size();
	
	// Set up the ann architecture
	for (int layerNum = 0; layerNum < numOfLayers; ++layerNum) {
		// Push an empty layer
		nn_layers.push_back(Layer());
		cout << "\nLayer: " << layerNum+1 << endl << "---------------\n";
		
		// Number of input connections to be added to each neuron
		int numOfInputs = layerNum == 0 ? 0 : shape[layerNum-1];
		
		// Add neurons to each Layer
		for (int n = 0; n < shape[layerNum]; ++n) {
			nn_layers.back().push_back(Neuron(numOfInputs, n));
			cout << "Added neuron " << n+1 << endl;
		}
	}
}

void NeuralNet::forwardProp(const vector<double> &input, double &output)
{	
	// Check if num. of input features = no. input neurons
	assert(input.size() == nn_layers[0].size());
	
	// Pass the input values through input neurons
	for (int i = 0; i < input.size(); ++i) {
		nn_layers[0][i].setOutput(input[i]);
	}

	// Forward propagation
	for (int layerNum = 1; layerNum < nn_layers.size(); ++layerNum) {
		Layer &prevLayer = nn_layers[layerNum -1];
		string activation = layerNum == nn_layers.size() - 1 ? "sigmoid" : "tanh";
		for (int n = 0; n < nn_layers[layerNum].size(); ++n) {
			nn_layers[layerNum][n].feedForward(prevLayer, activation);
		}
	}

	// Get predicted output: y'
	output = nn_layers.back()[0].getOutput();
}

void NeuralNet::backProp(const double &target)
{	
	int numOfLayers = nn_layers.size();
	Layer &outputLayer = nn_layers.back();

	// Get previous Layer
	// To calculate the gradients, each neuron needs output values from prev. layer's neurons
	// and weights from output connections.
	Layer &prevLayer_L = nn_layers[numOfLayers-2];
	
	// Calculate output layer gradients
	outputLayer[0].calcOutputGradients(target, prevLayer_L, "sigmoid");

	// Calculate hidden layer gradients
	for (int layerNum = numOfLayers - 2; layerNum > 0; --layerNum) {
		Layer &hiddenLayer = nn_layers[layerNum];
		Layer &nextLayer = nn_layers[layerNum+1];
		Layer &prevLayer_l = nn_layers[layerNum-1];
		for (int n = 0; n < hiddenLayer.size(); ++n) {
			hiddenLayer[n].calcHiddenGradients(nextLayer, prevLayer_l, "tanh");
		}
	}

	// Update all weights
	for (int layerNum = 1; layerNum < numOfLayers; ++layerNum) {
		Layer &layer = nn_layers[layerNum];
		for (int n = 0; n < layer.size(); ++n) {
			layer[n].updateWeights();
		}
	}
}

double NeuralNet::getLoss(const double &output, const double &target)
{
	double loss = target == 1 ? -log(output) : -log(1 - output);
	return loss;
}

/*---------------------- Main function--------------------------*/

int main()
{
    vector<int> shape;
    shape.push_back(2);
    shape.push_back(4);
    shape.push_back(1);

    NeuralNet xorNet(shape);

    vector<double> input, losses;
    double output, target;

    TrainingData td("./data/training-data-xor.csv");
    
    for (int t = 0; t < 10; ++t) { // Epochs
        int m = 0;
        double loss = 0.0;

        cout << "\n-----------------------------------------------" << endl;

        while (!td.isEOF()) {
            ++m;
            cout << "\n[Epoch: " << t+1 << ", Training sample: " << m+1 << "] " << endl;

            td.getTrainingData(input, target);
            cout << "Input: " << input[0] << " " << input[1] << endl;

            xorNet.forwardProp(input, output); // Forward propagation 
            
            cout << "Output: " << output << endl;
            cout << "Target: " << target << endl;

            loss += xorNet.getLoss(output, target); // Calculate loss over the entire training samples

            xorNet.backProp(target); // Back propagation
        }

        loss /= m;
        losses.push_back(loss);
        
        td.startOver();
    }

    td.close();

    cout << "-----------------------------------------------" << endl;
    cout << "\nLoss trend after each epoch:" << endl;
    cout << "----------------------------" << endl;
    for (int i = 0; i < losses.size(); ++i) {
        cout << losses[i] << endl;
    }
}