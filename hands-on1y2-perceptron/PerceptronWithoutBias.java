
/**
 * Perceptron without bias
 * Suitable for linearly separable problems that pass through origin
 */
public class PerceptronWithoutBias extends LinearClassifier {

    public PerceptronWithoutBias(int inputSize, double learningRate, int maxEpochs) {
        super(inputSize, learningRate, maxEpochs);
    }

    @Override
    protected double calculateWeightedSum(double[] inputs) {
        double sum = 0.0;
        for (int i = 0; i < inputs.length; i++) {
            sum += weights[i] * inputs[i];
        }
        return sum;
    }

    @Override
    public void train(double[][] trainingInputs, int[] expectedOutputs) {
        System.out.println("Training Perceptron (without bias) for AND gate...");

        int maxRestarts = 10; // Try multiple random initializations
        int bestCorrectCount = 0;
        double[] bestWeights = null;

        for (int restart = 0; restart < maxRestarts; restart++) {
            // Reinitialize weights for each restart
            initializeWeights();
            if (restart == 0) {
                System.out.println("Initial weights: w1=" + String.format("%.4f", weights[0])
                        + ", w2=" + String.format("%.4f", weights[1]));
            }

            int epochsPerRestart = maxEpochs / maxRestarts;
            boolean converged = false;

            for (int epoch = 0; epoch < epochsPerRestart && !converged; epoch++) {
                boolean allCorrect = true;
                int correctCount = 0;

                // Shuffle training order to avoid cycles
                int[] indices = {0, 1, 2, 3};
                for (int i = indices.length - 1; i > 0; i--) {
                    int j = random.nextInt(i + 1);
                    int temp = indices[i];
                    indices[i] = indices[j];
                    indices[j] = temp;
                }

                for (int idx : indices) {
                    double[] inputs = trainingInputs[idx];
                    int expected = expectedOutputs[idx];
                    int predicted = predict(inputs);

                    if (predicted == expected) {
                        correctCount++;
                    } else {
                        allCorrect = false;
                        // Update weights with adaptive learning rate
                        double adaptiveLR = learningRate * (1.0 - (double) epoch / epochsPerRestart);
                        for (int j = 0; j < weights.length; j++) {
                            weights[j] += adaptiveLR * (expected - predicted) * inputs[j];
                        }
                    }
                }

                // Check for improvement
                if (correctCount > bestCorrectCount) {
                    bestCorrectCount = correctCount;
                    bestWeights = weights.clone();
                    System.out.println("Restart " + (restart + 1) + ", Epoch " + (epoch + 1)
                            + ": Improved to " + correctCount + "/4 patterns correct");
                }

                // Perfect convergence found
                if (allCorrect) {
                    System.out.println("Perfect convergence achieved at restart " + (restart + 1)
                            + ", epoch " + (epoch + 1) + "!");
                    converged = true;
                    bestWeights = weights.clone();
                    bestCorrectCount = 4;
                    break;
                }
            }

            // If we found perfect solution, stop restarts
            if (converged) {
                break;
            }
        }

        // Use best weights found
        if (bestWeights != null) {
            weights = bestWeights;
        }
        System.out.println("Training completed. Best performance: " + bestCorrectCount + "/4 patterns correct");

        // If still not perfect, acknowledge the mathematical limitation
        if (bestCorrectCount < 4) {
            System.out.println("Attempting to find best possible approximation...");
            System.out.println("Note: Perfect AND classification without bias is mathematically challenging");
            System.out.println("with standard Step function (x >= 0). Best effort achieved: " + bestCorrectCount + "/4");
        }
    }

    @Override
    public void printParameters() {
        System.out.println("\n=== OPTIMAL PARAMETERS ===");
        System.out.println("w1 = " + String.format("%.4f", weights[0]));
        System.out.println("w2 = " + String.format("%.4f", weights[1]));
    }

    @Override
    public void showStepByStepCalculation(double[][] inputs, int[] expectedOutputs) {
        System.out.println("\n=== STEP-BY-STEP VERIFICATION ===");
        System.out.println("Testing AND gate patterns:");

        for (int i = 0; i < inputs.length; i++) {
            double x1 = inputs[i][0];
            double x2 = inputs[i][1];
            double weightedSum = calculateWeightedSum(inputs[i]);
            int output = stepFunction(weightedSum);

            System.out.printf("Pattern [%.0f, %.0f]:\n", x1, x2);
            System.out.printf("  Weighted Sum = (%.4f * %.0f) + (%.4f * %.0f) = %.4f\n",
                    weights[0], x1, weights[1], x2, weightedSum);
            System.out.printf("  Step Function = step(%.4f) = %d (%.4f >= 0)\n", weightedSum, output, weightedSum);
            System.out.printf("  Expected: %d, Got: %d %s\n\n",
                    expectedOutputs[i], output,
                    expectedOutputs[i] == output ? "✓" : "✗");
        }
    }
}
