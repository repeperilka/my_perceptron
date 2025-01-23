using System.IO;

public class Perceptron{

    public Layer[] layers;

    public Perceptron(){
        layers = new Layer[0];
    }
    public Perceptron(int[] _layers, ActivationFunction _function){
        layers = new Layer[_layers.Length];
        for(int i = 0; i < _layers.Length; i++){
            layers[i] = new Layer(i == 0? 0 : _layers[i - 1], _layers[i], _function, new Random());
        }
    }
    public void Train(DataPoint[] _dataBatch, double _learningRate, double _momentum){
        for(int i = 0; i < _dataBatch.Length; i++){

            double[] result = ForwardsPropagation(_dataBatch[i].input);
            BackwardsPropagation(result, _dataBatch[i].expectedOutput);

            if(i == _dataBatch.Length - 1){
                double error = 0;
                for(int e = 0; e < result.Length; e++){
                    error += Cost(result[e], _dataBatch[i].expectedOutput[e]);
                }
                error /= result.Length;
                error *= 100f;

                Console.WriteLine("Cost = " + error);
            }
        }


        for(int i = 0; i < layers.Length; i++){
            layers[i].ApplyGradients(_learningRate, _momentum);
        }
    }

    public double[] ForwardsPropagation(double[] _input){
        double[] input = _input;
        layers[0].SetValues(_input);
        for(int i = 1; i < layers.Length; i++){
            double[] output = layers[i].GetValues(input);
            input = output;
        }
        return input;
    }
    public void BackwardsPropagation(double[] _results, double[] _expected){
        double[] costResult = new double[_results.Length];
        for(int i = 0; i < costResult.Length; i++){
            costResult[i] = DerivativeCost(_results[i], _expected[i]);
        }

        layers[layers.Length - 1].CalculateOutputLayerNodeValues(_expected);
        layers[layers.Length - 1].UpdateGradients(layers[layers.Length - 2].nodes);

        for(int i = layers.Length - 2; i > 0; i--){
            layers[i].CalculateHiddenLayerNodeValues(layers[i + 1]);
            layers[i].UpdateGradients(layers[i - 1].nodes);
        }
    }


    public double Cost(double _output, double _expected){
        double cost = _output - _expected;
        return cost * cost;
    }
    public double DerivativeCost(double _output, double _expected){
        double cost = _output - _expected;
        return 2 * cost;
    }
}
public class Layer{
    public int nodeAmountPrev;
    public int nodeAmount;
    public double[] nodes;
    public double[] weightedNodes;
    public double[] nodeValues;
    
    public double[][] weights;
    public double[][] weightsCost;
    public double[][] prevWeightCost;
    
    public double[] bias;
    public double[] biasCost;
    public double[] prevBiasCost;
    public ActivationFunction activationFunc;

    public Layer(){
        nodes = new double[0];
        weightedNodes = new double[0];
        nodeValues = new double[0];
        bias = new double[0];
        biasCost = new double[0];
        weights = new double[0][];
        weightsCost = new double[0][];
    }
    //new layer
    public Layer(int _previousNodes, int _nodeAmount, ActivationFunction _activation, Random rand){
        nodeAmountPrev = _previousNodes;
        nodeAmount = _nodeAmount;
        activationFunc = _activation;


        //setting up bias
        bias = new double[nodeAmount];
        for(int i = 0; i < nodeAmount; i++){
            bias[i] = rand.NextDouble();
        }

        //setting up nodes
        nodes = new double[nodeAmount];

        //setting up weights
        weights = new double[nodeAmountPrev][];
        for(int i = 0; i < weights.Length; i++){
            weights[i] = new double[nodeAmount];
            for(int e = 0; e < nodeAmount; e++){
                double random = rand.NextDouble() * 2 - 1;
                weights[i][e] = random / Math.Sqrt(weights.Length);
            }
        }

        InitializeCosts();
    }

    //load layer
    public Layer(double[][] _weights, double[] _bias){
        nodeAmountPrev = _weights.Length;
        nodeAmount = _weights[0].Length;

        bias = _bias;
        weights = _weights;
        if(_weights.Length != 0){
            nodes = new double[_weights[0].Length];
        }else{
            nodes = new double[0];
        }

        InitializeCosts();
    }

    public void InitializeCosts(){
        weightsCost = new double[nodeAmountPrev][];
        for(int i = 0; i < nodeAmountPrev; i++){
            weightsCost[i] = new double[nodeAmount];
        }

        biasCost = new double[nodeAmount];
        
    }

    public double[] GetValues(double[] _previousValues){
        if(_previousValues.Length != weights.Length){
            Console.WriteLine("Node input amount not match in GetValues()");
            return new double[nodes.Length];
        }



        weightedNodes = new double[nodes.Length];

        for(int e = 0; e < nodeAmountPrev; e++){
            for(int i = 0; i < nodeAmount; i++){
                weightedNodes[i] += (_previousValues[e] * weights[e][i]);
            }
        }
        for(int i = 0; i < nodes.Length; i++){
            weightedNodes[i] += bias[i];
            nodes[i] = Activation(weightedNodes[i]);
        }
        return nodes;
    }
    public double[] SetValues(double[] _values){
        if(_values.Length != nodeAmount)
        {
            Console.WriteLine("SetValues mismach array length");
            return _values;
        }
        nodes = _values;
        return nodes;
    }

	public double[] CalculateHiddenLayerNodeValues(Layer _nextLayer)
	{
        nodeValues = new double[nodeAmount];
		for (int thisNode = 0; thisNode < nodeAmount; thisNode++)
		{
			double thisNodeValue = 0;
			for (int nextNode = 0; nextNode < _nextLayer.nodeValues.Length; nextNode++)
			{
				// Partial derivative of the weightedsum / activation equals the next layers weight
				double weightedInputDerivative = _nextLayer.weights[thisNode][nextNode];

                //sum of the partial derivatives of every nodeValue it influences
				thisNodeValue += weightedInputDerivative * _nextLayer.nodeValues[nextNode];
			}
            //multiplies the derivative of the current's layer activation by the derivative of the activation / weighted sum
            //to chain them
			thisNodeValue *= ActivationDerivative(weightedNodes[thisNode]);

            //stores weighedsum derivative in nodeValues
            nodeValues[thisNode] = thisNodeValue;
		
        }
        return nodeValues;

	}
	public double[] CalculateOutputLayerNodeValues(double[] _expectedOutputs)
	{
        double DerivativeCost(double _output, double _expected){
            return 2f * (_output - _expected);
        }

        nodeValues = new double[nodeAmount];
		for (int i = 0; i < nodeAmount; i++)
		{
			// Evaluate partial derivatives for current node: cost/activation & activation/weightedInput
			double costDerivative = DerivativeCost(nodes[i], _expectedOutputs[i]);
			double activationDerivative = ActivationDerivative(weightedNodes[i]);
			nodeValues[i] = costDerivative * activationDerivative;
		}
        return nodeValues;
	}


    public void UpdateGradients(double[] _previousActivations)
	{
        
			for (int thisNode = 0; thisNode < nodeAmount; thisNode++)
			{
				double nodeValue = nodeValues[thisNode];
				for (int nextNode = 0; nextNode < nodeAmountPrev; nextNode++)
				{
                    //partial derivative weight / weightedsum is the previous activation
					double derivativeCostWrtWeight = _previousActivations[nextNode];
                    //multiply by this layer's node value to chain partial derivatives
                    derivativeCostWrtWeight *= nodeValues[thisNode];
                    
                    //store derivatives per batch for gradiant descent
					weightsCost[nextNode][thisNode] += derivativeCostWrtWeight;
				}
			}

			for (int thisNode = 0; thisNode < nodeAmount; thisNode++)
			{
				//partial derivative weightedsum / bias equals 1
				double derivativeCostWrtBias = 1f;
                //multiply with nodeValues to chain partial derivatives
                derivativeCostWrtBias *= nodeValues[thisNode];

                //store derivatives per batch for gradiant descent
				biasCost[thisNode] += derivativeCostWrtBias;
			}
	}
	public void ApplyGradients(double learnRate, double momentum)
	{
		for (int i = 0; i < nodeAmountPrev; i++)
		{
            for(int e = 0; e < nodeAmount; e++){
                double weight = weights[i][e];
                weights[i][e] = weight - weightsCost[i][e] * learnRate;
                weightsCost[i][e] = weightsCost[i][e] * momentum;
            }
		}


		for (int i = 0; i < bias.Length; i++)
		{
            bias[i] = bias[i] - biasCost[i] * learnRate;
            biasCost[i] = biasCost[i] * momentum; 
		}
	}



    public double Activation(double _value){
        switch(activationFunc){
            case ActivationFunction.ReLU:
                return Math.Max(0, _value);
            case ActivationFunction.Sigmoid:
                return 1f / (1f + Math.Exp(-_value));
            default:
                return 1f / (1f + Math.Exp(-_value));
        }
    }
    public double ActivationDerivative(double _value){
        switch(activationFunc){
            case ActivationFunction.ReLU:
                return _value > 0 ? 1f : 0f;
            case ActivationFunction.Sigmoid:
                double activationSigmoid = Activation(_value);
                return activationSigmoid * (1f - activationSigmoid);
            default:
                double activation = Activation(_value);
                return activation * (1f - activation);
        }
    }

}
public enum ActivationFunction{
    Sigmoid,
    ReLU
}