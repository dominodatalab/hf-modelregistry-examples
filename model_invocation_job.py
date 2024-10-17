import mlflow
import mlflow.pyfunc
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch
import argparse

def invoke_model(model_name,model_version,input):
    test_input = [input]    
    model_uri = f"models:/{model_name}/{model_version}"    
    loaded_model = mlflow.pyfunc.load_model(model_uri)
    # Invoke the model by passing input data
    predictions = loaded_model.predict(test_input)
    # Print the predictions
    print(f"Predicted classes: {predictions}")
    predictions = loaded_model.predict(test_input)
    print(f"Predictions: {predictions}")
    return predictions


# Standard Python entry point for running the script
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Log and invoke a Hugging Face model using MLflow.")
    
    # Add command-line arguments
    parser.add_argument("--model_name", type=str, required=True, help="MLflow Registered Model Name")
    parser.add_argument("--model_version", type=int, required=True, help="MLflow Registered Model Version")
    parser.add_argument("--text_input", type=str, required=True, help="Test Input")
    # Parse the arguments from the command line
    args = parser.parse_args()
    invoke_model(args.model_name,args.model_version,args.text_input)
    