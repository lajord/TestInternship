
# AI agent technical model

The aim of this project is to create a specialized conversational agent to help customers by answering their questions from a database of documents.

For this project, I used the context of a bank advisor


**Requirements**

- python 3.x
- CUDA (if you have nvdia GPU)


## Ollama

You have to install ollama on your desktop
go on the website of ollama : ollama.com

When ollama is installed go to your terminal and install deepseek version 7b or higher if your pc has the capacity.

```bash
ollama run deepseek-r1:7b
```
## Firts : Indexation.py

#### Features
- Q&A extraction from JSON and TXT files.
- Generate embeddings with **DeepSeek**.
- Store and search embeddings with **ChromaDB**.
- Automatic indexing of all JSON and TXT files in a directory.

Installing dependencies for the python library
```bash
pip install ollama chromadb
```
Make sure that ollama is installed and correctly configured on your machine to run the deepseek model.

#### Use

**You have already documentation on the github, the folder name is DocumentationIndexation, so if all dependencies are install you can just execute the scipt.
It will be create a folder name chroma_db**


**1.Indexing a single file**

You can index a JSON or TXT file with the following python line:
```bash
index_with_deepseek_chroma("chemin/vers/fichier.json")
```
**2.Indexing a complete directory**

To index all JSON and TXT files in a :
```bash
process_directory("chemin/vers/repertoire")
```

#### File structure
- JSON: Each file must contain a list of objects with question and answer fields.
```bash
[
  { "question": "Comment ouvrir un compte bancaire ?", "answer": "Vous pouvez ouvrir un compte en ligne ou en agence..." },
  { "question": "Quels sont les frais de transfert ?", "answer": "Les frais de transfert varient selon..." }
]
```
- TXT: Each question and answer or other structure must be separated by a blank line.
```bash
Comment ouvrir un compte bancaire ?
Vous pouvez ouvrir un compte en ligne ou en agence...

Quels sont les frais de transfert ?
Les frais de transfert varient selon...
```
## Second : Finetuning.ipynb

The purpose of this notebook is to fine tune the model **Phi-3-mini-4k-instruct** with a DataSet

#### Requirements:
- torch
- unsloth
- datasets 
- transformers
- trl
- peft
- accelerate
- bitsandbytes 
- packaging 
- ninja 
- einops 
- flash-attn
- xformers 



If you use google colab beacause you have no nvdia GPU, The dependencies to install are the following and are to be put at the beginning of your code on google colab

```bash
%%capture
import torch
major_version, minor_version = torch.cuda.get_device_capability()
!pip install "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"
if major_version >= 8:
    !pip install --no-deps packaging ninja einops flash-attn xformers trl peft accelerate bitsandbytes
else:
    !pip install --no-deps xformers trl peft accelerate bitsandbytes
pass
```

For local utilisation :

**Installing dependencies**
```bash
pip install "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"
```
- If yout GPU is compatible with CUDA 8 or plus
```bash
pip install --no-deps packaging==23.2 ninja==1.11.1 einops==0.6.1 flash-attn==2.3.0 xformers==0.0.22 trl==0.7.6 peft==0.8.2 accelerate==0.25.0 bitsandbytes==0.41.1
```
- else
```bash
pip install --no-deps xformers==0.0.22 trl==0.7.6 peft==0.8.2 accelerate==0.25.0 bitsandbytes==0.41.1
```
- Install Pytorch
```bash
pip install --upgrade torch==2.6.0+cu124 torchvision==0.17.0+cu124 torchaudio==2.6.0+cu124 --extra-index-url https://download.pytorch.org/whl/cu124
```

**To run this notebook you need a dataset, in the github there is already a dataset named train_data.jsonl, so when you have installed all the dependencies you can run the notebook.
At the end of the script, you need to retrieve the .gguf file. .**
## Third : Custom Model with Ollama

After the fine tuning and recup your model, if you have not already done so, you must rename it: unsloth.Q4_K_M.gguf.

Create a personal model with Ollama.

- You should have the Modelfile file in your folder and it should look like this  
```bash
FROM ./unsloth.Q4_K_M.gguf

SYSTEM You are a professional bank advisor assistant. Your sole responsibility is to answer questions related exclusively to banking information—such as fees, account details, and available services. You must always respond in English, regardless of the language used by the user. If a question is unrelated to banking or if you do not have enough information to provide a correct answer, do not respond with any content. Instead, instruct the user to contact a banker for further assistance.

```
- To create the model with ollama you need to do
```bash
ollama create model_finetune_3 -f ./Modelfile

```
It's important to name the model, model_finetune_3 


## Last : Chat.py

#### So let's summarize 
- You have indexed the documents using indexation.py and a chroma_db file has been created in your directory.
- Using the notebook finetuning.ipynb you have fine tuned the model Phi-3-mini-4k using a dataset (train_data.jsonl) and you have recovered the model which is a .gguf.
- You have created the model , model_finetune_3 with the help of Ollama

**Now all you have to do is run the chat.py script and interact with the model.**