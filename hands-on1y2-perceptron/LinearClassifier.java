
import java.util.Random;

/**
 * Abstract base class for Linear Classifiers Represents a perceptron with
 * common functionality
 */
public abstract class LinearClassifier {

    protected double[] weights;
    protected double learningRate;
    protected Random random;
    protected int maxEpochs;

    public LinearClassifier(int inputSize, double learningRate, int maxEpochs) {
        this.weights = new double[inputSize];
        this.learningRate = learningRate;
        this.maxEpochs = maxEpochs;
        this.random = new Random();
        initializeWeights();
    }

    /**
     * Initialize weights randomly between -0.5 and 0.5
     */
    protected void initializeWeights() {
        for (int i = 0; i < weights.length; i++) {
            weights[i] = random.nextDouble() - 0.5;
        }
    }

    /**
     * Step activation function Returns 1 if input >= 0, otherwise 0
     */
    protected int stepFunction(double input) {
        return input >= 0 ? 1 : 0;
    }

    /**
     * Calculate weighted sum of inputs
     */
    protected abstract double calculateWeightedSum(double[] inputs);

    /**
     * Predict output for given inputs
     */
    public int predict(double[] inputs) {
        double weightedSum = calculateWeightedSum(inputs);
        return stepFunction(weightedSum);
    }

    /**
     * Train the perceptron with given training data
     */
    public abstract void train(double[][] trainingInputs, int[] expectedOutputs);

    /**
     * Print weights and bias (if applicable)
     */
    public abstract void printParameters();

    /**
     * Show step-by-step calculation for verification
     */
    public abstract void showStepByStepCalculation(double[][] inputs, int[] expectedOutputs);
}
