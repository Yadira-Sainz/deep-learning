
/**
 * Perceptron with bias for OR gate classification
 * Uses standard perceptron learning algorithm with random initialization
 */
public class PerceptronWithBias extends LinearClassifier {

    private double bias; // Bias parameter (b)

    public PerceptronWithBias(int inputSize, double learningRate, int maxEpochs) {
        super(inputSize, learningRate, maxEpochs);
        // Initialize bias randomly as per specification 3b
        this.bias = random.nextGaussian() * 0.5;
    }

    @Override
    protected int stepFunction(double input) {
        return input > 0 ? 1 : 0;
    }

    @Override
    protected double calculateWeightedSum(double[] inputs) {
        double sum = bias; // Start with bias
        for (int i = 0; i < inputs.length; i++) {
            sum += weights[i] * inputs[i];
        }
        return sum;
    }

    @Override
    public boolean train(double[][] trainingInputs, int[] expectedOutputs) {
        // Standard perceptron learning algorithm with bias
        for (int epoch = 0; epoch < maxEpochs; epoch++) {
            boolean allCorrect = true;

            for (int i = 0; i < trainingInputs.length; i++) {
                double[] inputs = trainingInputs[i];
                int expected = expectedOutputs[i];
                int predicted = predict(inputs);

                if (predicted != expected) {
                    allCorrect = false;
                    int error = expected - predicted;

                    // Update weights: w = w + α * error * input
                    for (int j = 0; j < weights.length; j++) {
                        weights[j] += learningRate * error * inputs[j];
                    }

                    // Update bias: b = b + α * error * 1
                    bias += learningRate * error;
                }
            }

            if (allCorrect) {
                // Perfect convergence achieved
                return true;
            }
        }

        // Training completed (should converge for OR gate)
        return true;
    }

    @Override
    public void printParameters() {
        // Print ONLY what specifications require: w1, w2, and bias (b)
        System.out.println("=".repeat(60));
        System.out.println("PERCEPTRON WITH BIAS - OR GATE:");
        System.out.println("=".repeat(60));
        System.out.println("w1 = " + String.format("%.4f", weights[0]));
        System.out.println("w2 = " + String.format("%.4f", weights[1]));
        System.out.println("b = " + String.format("%.4f", bias));
        System.out.println(); // Line break after parameters to separate from patterns
    }

    @Override
    public void showStepByStepCalculation(double[][] inputs, int[] expectedOutputs) {
        // Print ONLY step-by-step calculations: Weighted Sum and Activation Function
        for (int i = 0; i < inputs.length; i++) {
            double x1 = inputs[i][0];
            double x2 = inputs[i][1];
            double weightedSum = calculateWeightedSum(inputs[i]);
            int output = stepFunction(weightedSum);

            System.out.printf("Pattern [%.0f, %.0f]:\n", x1, x2);
            System.out.printf("Weighted Sum = (%.4f * %.0f) + (%.4f * %.0f) + %.4f = %.4f\n",
                    weights[0], x1, weights[1], x2, bias, weightedSum);
            System.out.printf("Activation Function = step(%.4f) = %d\n", weightedSum, output);
            System.out.println();
        }
    }
}
