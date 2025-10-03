# Perceptron Implementation - Linear Classifiers

## Project Overview

This project implements a complete perceptron learning system using object-oriented design principles. It demonstrates the fundamental concepts of neural networks through linear classification of logical gates (AND, OR) with both Perceptron without bias and Perceptron with bias implementations.

## UML Class Diagram Design

### Class Hierarchy:

```
LinearClassifier (Abstract)
├── PerceptronWithoutBias (without bias)
└── PerceptronWithBias (with bias)
```

### UML Class Diagram:

```
┌─────────────────────────────────────────────┐
│             LinearClassifier                │
│               <<abstract>>                  │
├─────────────────────────────────────────────┤
│ # weights: double[]                         │
│ # learningRate: double                      │
│ # random: Random                            │
│ # maxEpochs: int                            │
├─────────────────────────────────────────────┤
│ + LinearClassifier(int, double, int)        │
│ # initializeWeights(): void                 │
│ # stepFunction(double): int                 │
│ # getActivationThreshold(): double          │
│ # calculateWeightedSum(double[]): double    │
│   <<abstract>>                              │
│ + predict(double[]): int                    │
│ + train(double[][], int[]): boolean         │
│   <<abstract>>                              │
│ + printParameters(): void <<abstract>>      │
│ + showStepByStepCalculation(double[][],     │
│   int[]): void <<abstract>>                 │
└─────────────────────────────────────────────┘
                        △
                        │
            ┌───────────┴───────────┐
            │                       │
┌───────────────────────────┐ ┌─────────────────────────────┐
│   PerceptronWithoutBias   │ │    PerceptronWithBias       │
├───────────────────────────┤ ├─────────────────────────────┤
│                           │ │ - bias: double              │
├───────────────────────────┤ ├─────────────────────────────┤
│ + PerceptronWithoutBias   │ │ + PerceptronWithBias        │
│   (int, double, int)      │ │   (int, double, int)        │
│ # stepFunction(double):   │ │ # stepFunction(double): int │
│   int                     │ │ # calculateWeightedSum      │
│ # calculateWeightedSum    │ │   (double[]): double        │
│   (double[]): double      │ │ + train(double[][], int[]): │
│ + train(double[][],       │ │   boolean                   │
│   int[]): boolean         │ │ + printParameters(): void   │
│ + printParameters(): void │ │ + showStepByStep            │
│ + showStepByStep          │ │   Calculation(double[][],   │
│   Calculation(double[][],  │ │   int[]): void              │
│   int[]): void            │ └─────────────────────────────┘
└───────────────────────────┘
```

## Implementation Details

### 1. LinearClassifier (Abstract Base Class)

- **Purpose**: Provides the abstraction for all linear classifiers following OOP principles
- **Key Features**:
  - Weight initialization with random values between -1.0 and 1.0
  - Step activation function (returns 1 for input > threshold, 0 otherwise)
  - Configurable activation threshold (default 0.0)
  - Abstract methods for specialized implementations
- **Design Pattern**: Template Method Pattern
- **Abstraction Level**: Defines the contract for any linear classifier

### 2. PerceptronWithoutBias (Without Bias - Concrete Implementation)

- **Purpose**: Implements basic perceptron without bias term
- **Use Case**: AND gate classification (linearly separable through origin)
- **Formula**: output = step(∑(wᵢ × xᵢ))
- **Learning Rule**: wᵢ(new) = wᵢ(old) + η × (target - output) × xᵢ
- **Limitation**: Can only classify patterns that are linearly separable through the origin

### 3. PerceptronWithBias (With Bias - Concrete Implementation)

- **Purpose**: Implements complete Linear Classifier with bias term
- **Use Case**: OR gate classification (requires offset from origin)
- **Formula**: output = step(∑(wᵢ × xᵢ) + bias)
- **Learning Rules**:
  - wᵢ(new) = wᵢ(old) + η × (target - output) × xᵢ
  - bias(new) = bias(old) + η × (target - output)
- **Advantage**: Can classify any linearly separable pattern

## Object-Oriented Abstraction Process

### 1. **Identification of Common Behaviors**

- All linear classifiers need weight management
- All require a prediction mechanism
- All use similar training patterns
- All need parameter display capabilities

### 2. **Abstraction Extraction**

- **LinearClassifier**: Captures the essence of any linear classification algorithm
- Defines the interface without implementation details
- Provides common functionality through protected methods

### 3. **Concrete Implementations**

- **PerceptronWithoutBias**: Basic implementation without bias for AND gate
- **PerceptronWithBias**: Enhanced implementation with bias for OR gate
- Both inherit common behavior from abstract parent
- Each implements abstract methods with specific logic

### 4. **Polymorphism Benefits**

- Can treat any linear classifier uniformly
- Easy to extend with new classifier types
- Maintains consistency across implementations

## Training Data

### AND Gate Truth Table (for PerceptronWithoutBias):

| Input 1 (x₁) | Input 2 (x₂) | Expected Output | Description |
| ------------ | ------------ | --------------- | ----------- |
| 0            | 0            | 0               | Both false  |
| 0            | 1            | 0               | One true    |
| 1            | 0            | 0               | One true    |
| 1            | 1            | 1               | Both true   |

### OR Gate Truth Table (for PerceptronWithBias):

| Input 1 (x₁) | Input 2 (x₂) | Expected Output | Description       |
| ------------ | ------------ | --------------- | ----------------- |
| 0            | 0            | 0               | Both false        |
| 0            | 1            | 1               | At least one true |
| 1            | 0            | 1               | At least one true |
| 1            | 1            | 1               | Both true         |

## Project Structure

```
hands-on1y2-perceptron/
├── README.md
├── run.sh
├── LinearClassifier.java
├── PerceptronWithoutBias.java
├── PerceptronWithBias.java
├── Main.java
└── results/
    └── training_logs.txt (generated after execution)
```

## Usage

### Quick Start:

```bash
# Make script executable (first time only)
chmod +x run.sh

# Compile and run
./run.sh
```

### Manual Compilation and Execution:

```bash
# Compile all Java files
javac *.java

# Run the demonstration
java Main

# Clean compiled files
rm *.class
```

### Configuration Parameters:

- **Learning Rate (η)**: Various rates tested automatically (0.001 to 1.5)
- **Maximum Epochs**: Up to 20,000 (adaptive based on convergence)
- **Weight Initialization Range**: [-1.0, 1.0]
- **Bias Initialization Range**: Gaussian distribution with σ=0.5

## Expected Output

The program demonstrates:

1. **Training Results** (silent training process):

2. **Final Parameters**:

   ```
   ============================================================
   PERCEPTRON WITHOUT BIAS - AND GATE:
   ============================================================
   w1 = 0.4000
   w2 = 0.4000

   ============================================================
   PERCEPTRON WITH BIAS - OR GATE:
   ============================================================
   w1 = 0.3000
   w2 = 0.3000
   b = -0.1000
   ```

3. **Step-by-Step Verification**:

   ```
   Pattern [0, 0]:
   Weighted Sum = (0.4000 * 0) + (0.4000 * 0) = 0.0000
   Activation Function = step(0.0000) = 0

   Pattern [0, 1]:
   Weighted Sum = (0.4000 * 0) + (0.4000 * 1) = 0.4000
   Activation Function = step(0.4000) = 1

   Pattern [0, 0]:
   Weighted Sum = (0.3000 * 0) + (0.3000 * 0) + -0.1000 = -0.1000
   Activation Function = step(-0.1000) = 0
   ```

4. **Classification Results**:
   - AND gate (PerceptronWithoutBias): Trained with adaptive learning approach
   - OR gate (PerceptronWithBias): 100% accuracy (4/4 patterns correct)

## Key Learning Concepts

### 1. **Object-Oriented Design Principles**

- **Abstraction**: LinearClassifier captures essential linear classifier behavior
- **Inheritance**: PerceptronWithoutBias and PerceptronWithBias extend base functionality
- **Encapsulation**: Protected and private members ensure data integrity
- **Polymorphism**: Uniform treatment of different classifier types

### 2. **UML Design Benefits**

- **Clear Hierarchy**: Shows inheritance relationships
- **Method Signatures**: Defines interface contracts
- **Visibility Modifiers**: Proper encapsulation representation
- **Abstract Methods**: Clearly marked implementation requirements

### 3. **Machine Learning Fundamentals**

- **Supervised Learning**: Learning from labeled examples
- **Perceptron Learning Algorithm**: Weight adjustment based on errors
- **Convergence**: Algorithm stops when all patterns are correctly classified
- **Linear Separability**: Geometric interpretation of classification boundaries

### 4. **Neural Network Concepts**

- **Activation Functions**: Step function as simplest activation
- **Bias Term**: Critical for classification flexibility and offset capability
- **Learning Rate**: Impact on convergence speed and stability
- **Epochs**: Iterative learning process

## Troubleshooting

### Common Issues:

1. **Compilation Errors**: Ensure Java 8+ is installed and class names match files
2. **Permission Denied**: Run `chmod +x run.sh`
3. **No Convergence**: The implementation uses adaptive learning rates and multiple attempts
4. **ClassNotFoundException**: Verify all .java files are in same directory

### Performance Tips:

- The AND gate uses an enhanced retry mechanism with varied parameters
- OR gate typically converges quickly due to linear separability with bias
- Monitor the step-by-step calculations to verify correct learning

## Extensions and Improvements

### Possible Enhancements:

1. **Additional Logic Gates**: XOR, NAND, NOR implementations
2. **Visualization**: Plot decision boundaries and training progress
3. **Multiple Learning Rates**: Already implemented for AND gate
4. **Batch vs Online Learning**: Different training strategies
5. **Cross-Validation**: Model evaluation techniques

---

_This implementation demonstrates proper object-oriented abstraction for linear classifiers with PerceptronWithoutBias and PerceptronWithBias as concrete implementations._

- Verify input data format matches expected structure

## Extensions and Improvements

### Possible Enhancements:

1. **Additional Classifiers**: SVM, Logistic Regression extending ClasificadorLineal
2. **XOR Problem**: Demonstrate non-linear separability limitations
3. **Multiple Learning Rates**: Compare convergence behavior
4. **Batch vs Online Learning**: Different training strategies
5. **Cross-Validation**: Model evaluation techniques

---

_This implementation demonstrates proper object-oriented abstraction for linear classifiers with focus on the Perceptron and PerceptronConSesgo as concrete implementations._
