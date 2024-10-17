import mlflow
import mlflow.pyfunc
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch

class HuggingFaceModelWrapper(mlflow.pyfunc.PythonModel):
    def load_context(self, context):
        # Load the saved Hugging Face model and tokenizer
        self.model = AutoModelForSequenceClassification.from_pretrained(context.artifacts["model_path"])
        self.tokenizer = AutoTokenizer.from_pretrained(context.artifacts["model_path"])

    def predict(self, context, model_input):
        # Tokenize the input text
        inputs = self.tokenizer(model_input, return_tensors="pt", padding=True, truncation=True)

        # Perform inference with the model
        with torch.no_grad():
            outputs = self.model(**inputs)

        # Get the logits (raw model predictions)
        logits = outputs.logits

        # Convert logits to predicted class
        predicted_class = torch.argmax(logits, dim=1).numpy()

        return predicted_class
