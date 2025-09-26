# Perceptron Implementation - Linear Classifiers

## UML Class Diagram Design

### Class Hierarchy:

```
LinearClassifier (Abstract)
├── Perceptron (without bias)
└── PerceptronWithBias (with bias)
```

### UML Class Diagram:

```
┌─────────────────────────────────────┐
│           LinearClassifier          │
│               (Abstract)            │
├─────────────────────────────────────┤
│ # weights: double[]                 │
│ # learningRate: double              │
│ # random: Random                    │
│ # maxEpochs: int                    │
├─────────────────────────────────────┤
│ + LinearClassifier(int, double, int)│
│ # initializeWeights(): void         │
│ # stepFunction(double): int         │
│ # calculateWeightedSum(double[]): double (abstract) │
│ + predict(double[]): int            │
│ + train(double[][], int[]): void (abstract) │
│ + printParameters(): void (abstract)│
│ + showStepByStepCalculation(double[][], int[]): void (abstract) │
└─────────────────────────────────────┘
                    △
                    │
        ┌───────────┴───────────┐
        │                       │
┌───────────────────┐   ┌─────────────────────┐
│    Perceptron     │   │ PerceptronWithBias  │
├───────────────────┤   ├─────────────────────┤
│                   │   │ - bias: double      │
├───────────────────┤   ├─────────────────────┤
│ + Perceptron(int, │   │ + PerceptronWithBias│
│   double, int)    │   │   (int, double, int)│
│ + calculateWeighted│   │ + calculateWeighted │
│   Sum(double[]): │   │   Sum(double[]): │
│   double          │   │   double            │
│ + train(double[][], │   │ + train(double[][], │
│   int[]): void    │   │   int[]): void      │
│ + printParameters() │   │ + printParameters() │
│   : void          │   │   : void            │
│ + showStepByStep  │   │ + showStepByStep    │
│   Calculation(...) │   │   Calculation(...)  │
│   : void          │   │   : void            │
└───────────────────┘   │ + getBias(): double │
                        └─────────────────────┘
```

## Implementation Details

### 1. LinearClassifier (Abstract Base Class)
- **Purpose**: Provides common functionality for all linear classifiers
- **Key Features**:
  - Weight initialization with random values (-0.5 to 0.5)
  - Step activation function
  - Abstract methods for specific implementations

### 2. Perceptron (Without Bias)
- **Purpose**: Implements basic perceptron without bias term
- **Use Case**: AND gate classification
- **Formula**: output = step(w₁x₁ + w₂x₂)

### 3. PerceptronWithBias
- **Purpose**: Implements perceptron with bias term for better flexibility
- **Use Case**: OR gate classification  
- **Formula**: output = step(w₁x₁ + w₂x₂ + b)

## Training Data

### AND Gate Truth Table:
| Input 1 | Input 2 | Output |
|---------|---------|--------|
|    0    |    0    |   0    |
|    0    |    1    |   0    |
|    1    |    0    |   0    |
|    1    |    1    |   1    |

### OR Gate Truth Table:
| Input 1 | Input 2 | Output |
|---------|---------|--------|
|    0    |    0    |   0    |
|    0    |    1    |   1    |
|    1    |    0    |   1    |
|    1    |    1    |   1    |

## Usage

### Compilation and Execution:
```bash
# Make script executable
chmod +x run.sh

# Compile and run
./run.sh
```

### Manual Compilation:
```bash
javac *.java
java PerceptronDemo
```

## Expected Output

The program will show:
1. Training progress for both perceptrons
2. Optimal weight values (w₁, w₂)
3. Optimal bias value (for PerceptronWithBias)
4. Step-by-step calculations for each test pattern
5. Verification that both classifiers learned correctly

## Key Learning Concepts

1. **Object-Oriented Design**: Inheritance and abstraction
2. **Perceptron Algorithm**: Weight update rule
3. **Activation Functions**: Step function implementation
4. **Linear Separability**: AND vs OR gate classification
5. **Bias Term**: Impact on classification capability