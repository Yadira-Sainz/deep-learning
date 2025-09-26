/**
 * Main class to demonstrate Perceptron implementations
 * Tests AND gate with Perceptron (no bias) and OR gate with PerceptronWithBias
 */
public class PerceptronDemo {
    
    public static void main(String[] args) {
        System.out.println("=".repeat(60));
        System.out.println("PERCEPTRON DEMONSTRATION");
        System.out.println("=".repeat(60));
        
        // Test AND gate with Perceptron (no bias)
        testAndGate();
        
        System.out.println("\n" + "=".repeat(60));
        
        // Test OR gate with PerceptronWithBias
        testOrGate();
    }
    
    private static void testAndGate() {
        System.out.println("TESTING AND GATE WITH PERCEPTRON (NO BIAS)");
        System.out.println("-".repeat(60));
        
        // AND gate training data
        double[][] andInputs = {
            {0, 0},
            {0, 1},
            {1, 0},
            {1, 1}
        };
        int[] andExpected = {0, 0, 0, 1};
        
        // Create and train perceptron
        Perceptron andPerceptron = new Perceptron(2, 0.1, 1000);
        andPerceptron.train(andInputs, andExpected);
        
        // Show results
        andPerceptron.printParameters();
        andPerceptron.showStepByStepCalculation(andInputs, andExpected);
    }
    
    private static void testOrGate() {
        System.out.println("TESTING OR GATE WITH PERCEPTRON (WITH BIAS)");
        System.out.println("-".repeat(60));
        
        // OR gate training data
        double[][] orInputs = {
            {0, 0},
            {0, 1},
            {1, 0},
            {1, 1}
        };
        int[] orExpected = {0, 1, 1, 1};
        
        // Create and train perceptron with bias
        PerceptronWithBias orPerceptron = new PerceptronWithBias(2, 0.1, 1000);
        orPerceptron.train(orInputs, orExpected);
        
        // Show results
        orPerceptron.printParameters();
        orPerceptron.showStepByStepCalculation(orInputs, orExpected);
    }
}