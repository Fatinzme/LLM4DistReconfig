# LLM4DistReconfig
LLM4DistReconfig: A Fine-tuned Large Language Model for Power Distribution Network Reconfiguration [ [link](https://arxiv.org/abs/2501.14960) ]  
Accepted in NAACL 2025 Conference Main Track  

This is the source code to regenerate the results of our paper LLM4DistReconfig.  
LLM4DistReconfig is a finetuned Llama 3.1 model that is able to solve the grid reconfiguration task for power systems. 
It has been tested on sizes of various sizes individually and combined and has been evaluated on unseen datasets of sizes both in and out of distribution i.e. (between the sizes it was trained on and outside).

We have prepared tempalates of the python notebooks and the sh files we used to generate the files for training and evaluation as well as training and evaluating the model.
Our system is robust and automated which allows for easy finetuning of the model and its evaluation but also for easy modification of the code to adapt to your needs.

## Requirements
```
- torch
- accelerate
- bitsandbytes
- peft
- transformers
- trl
```

## Datasets
In order to generate the required datasets that will be used for training and evaluation you will need access to the csv files which can be found here [ [link](https://github.com/panaschristou/grid-datasets) ] and should then be added to the folder csv_files inside Dataset-Notebooks (if the folder does not exist, create it and add the csv files from the link inside).  

Inside the Dataset-Notebooks folder you will find the dataset-generation-script notebook which you would run to generate all the required files to train your model. Going through the notebook we show how to process each individual file by creating the prompt, the input and the ouput to the model, splitting the file into training, validation and testing as well any auxiliary files that may be needed. Upon creating the files for each csv file we also provide functions for combining them together into a single file that can be used to train the model on a dataset with varying network sizes.

In the same folder we provided a file that converts MATLAB files to the required format to be used with the dataset-generation-script for easy conversion since most power systems engineers use MATLAB and we want them to have a smoother experience using our system.

## Finetuning
In the folder Model-Notebooks we provide the templates to fine-tune  a model on our dataset (required you have completed generating the datasets). You will need to modify the sh file with the right path to the dataset that you will use and also for the model. The model could be a model for hugging face for initial fine tuning or your own model that you want to fine tune even more. You will need to specify where to save the fine tuned model as well as other hyper parameters.

For fine tuning you can modify hyper parameters like the learning rate but also which components of the custom loss you want to use like cycles loss, subgraphs loss and invalid edges loss. You can use a combination of each or all of them as well. Invalid edges loss penalizes invalid edges that appear in the output, subgraphs loss penalizes any subgraphs found in the output and cycles loss penalizes any cyles found in the output. 

## Evaluation
In the folder Model-Notebooks we also provide the templates to evaluate your model. You can evaluate the model by specifying the path to the fine tuned model (or a model from hugging face for comparing with your fine tuned model) and you can specify how many samples to use for evaluation. The results will be saved in the evaluations folder as responses in the form of text and the metrics in the form of csv. In the Dataset-Notebooks folder we have also included a script that you can step through to generate metrics which provides an alternative to queueing a job.
