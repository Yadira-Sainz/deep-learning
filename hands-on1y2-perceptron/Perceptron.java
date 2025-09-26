/**
 * Perceptron without bias
 * Suitable for linearly separable problems that pass through origin
 */
public class Perceptron extends LinearClassifier {
    
    public Perceptron(int inputSize, double learningRate, int maxEpochs) {
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
        System.out.println("Initial weights: w1=" + String.format("%.4f", weights[0]) + 
                          ", w2=" + String.format("%.4f", weights[1]));
        
        for (int epoch = 0; epoch < maxEpochs; epoch++) {
            boolean converged = true;
            
            for (int i = 0; i < trainingInputs.length; i++) {
                double[] inputs = trainingInputs[i];
                int expected = expectedOutputs[i];
                int predicted = predict(inputs);
                
                if (predicted != expected) {
                    converged = false;
                    // Update weights: w = w + α(expected - predicted) * input
                    for (int j = 0; j < weights.length; j++) {
                        weights[j] += learningRate * (expected - predicted) * inputs[j];
                    }
                }
            }
            
            if (converged) {
                System.out.println("Converged after " + (epoch + 1) + " epochs");
                break;
            }
        }
    }
    
    @Override
    public void printParameters() {
        System.out.println("\n=== OPTIMAL PARAMETERS ===");
        System.out.println("w1 = " + String.format("%.4f", weights[0]));
        System.out.println("w2 = " + String.format("%.4f", weights[1]));
        System.out.println("bias = 0.0000 (no bias)");
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
            System.out.printf("  Step Function = step(%.4f) = %d\n", weightedSum, output);
            System.out.printf("  Expected: %d, Got: %d %s\n\n", 
                            expectedOutputs[i], output, 
                            expectedOutputs[i] == output ? "✓" : "✗");
        }
    }
}