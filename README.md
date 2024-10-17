## Registering Hugging Face Models via Domino Datasets

This repo demonstrates how Hugging Face models can be registered via Domino Datasets

The example notebook demonstrates how to register as well as run the models 

TBD - Running the registered model as a model api endpoint

## Run the model invocation as a job

Run the invocation job as follows

```
python /mtn/code/model_invocation_job.py --model_name "distilbert-base-uncased-finetuned-sst-2-english" --model_version 6 --text_input "I liked the movie"
```
