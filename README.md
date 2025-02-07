# LLM4DistReconfig Code
LLM4DistReconfig: A Fine-tuned Large Language Model for Power Distribution Network Reconfiguration [[link](https://arxiv.org/abs/2501.14960)]  
Accepted in NACL 2025 Conference Main Track  

This is the source code to regenerate the results of our paper LLM4DistReconfig. 
LLM4DistReconfig is a finetuned Llama 3.1 that is able to solve the grid reconfiguration task for power systems. 
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
