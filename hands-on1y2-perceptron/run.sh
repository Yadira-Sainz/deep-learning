#!/bin/bash

echo "Compiling Java files..."
javac *.java

if [ $? -eq 0 ]; then
    echo "Compilation successful!"
    echo "Running Perceptron Demo..."
    echo ""
    java Main
else
    echo "Compilation failed!"
fi