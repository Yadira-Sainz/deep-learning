import java.util.Random;

/**
 * Clase base abstracta para Clasificadores Lineales (Perceptrón).
 * Proporciona funcionalidad común, inicialización de pesos aleatoria
 * y la función de activación Step.
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
     * Inicializa los pesos de forma aleatoria entre -1.0 y 1.0
     */
    protected void initializeWeights() {
        for (int i = 0; i < weights.length; i++) {
            // Inicializa los pesos con valores aleatorios entre -1.0 y 1.0
            weights[i] = (random.nextDouble() * 2.0) - 1.0;
        }
    }

    /**
     * Función de activación Step (Escalón)
     * Retorna 1 si input > umbral (0.0 por defecto), sino 0.
     */
    protected int stepFunction(double input) {
        return input > getActivationThreshold() ? 1 : 0;
    }

    /**
     * Obtiene el umbral de activación. Por defecto 0.0.
     */
    protected double getActivationThreshold() {
        return 0.0; // Umbral por defecto
    }

    /**
     * Calcula la suma ponderada de las entradas.
     */
    protected abstract double calculateWeightedSum(double[] inputs);

    /**
     * Predice la salida para las entradas dadas.
     */
    public int predict(double[] inputs) {
        double weightedSum = calculateWeightedSum(inputs);
        return stepFunction(weightedSum);
    }

    /**
     * Entrena el perceptrón con los datos de entrenamiento.
     * @return true si el perceptrón convergió completamente, false si alcanzó maxEpochs.
     */
    public abstract boolean train(double[][] trainingInputs, int[] expectedOutputs);

    /**
     * Imprime los parámetros óptimos (pesos y bias, si aplica).
     */
    public abstract void printParameters();

    /**
     * Muestra el cálculo paso a paso para la verificación.
     */
    public abstract void showStepByStepCalculation(double[][] inputs, int[] expectedOutputs);
}
