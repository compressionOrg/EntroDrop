import random
import numpy as np
import torch
from datasets import load_dataset, load_from_disk

def fix_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed) 



def get_wikitext2(nsamples, seed, seqlen, tokenizer):
    # Load train and test datasets
    traindata = load_dataset('wikitext', 'wikitext-2-raw-v1', split='train')
    testdata = load_dataset('wikitext', 'wikitext-2-raw-v1', split='test')

    random.seed(seed)

    # Generate training data samples
    trainloader = []
    for _ in range(nsamples):
        while True:
            # Randomly select a sample from the training data
            i = random.randint(0, len(traindata) - 1)
            # Tokenize the selected text (no longer join the entire dataset)
            trainenc = tokenizer(traindata[i]['text'], return_tensors='pt')

            # Ensure the sequence length is larger than `seqlen`
            if trainenc.input_ids.shape[1] > seqlen:
                break  # Continue if valid sequence length is found

        # Randomly select a window from the tokenized text
        start_idx = random.randint(0, trainenc.input_ids.shape[1] - seqlen - 1)
        end_idx = start_idx + seqlen
        inp = trainenc.input_ids[:, start_idx:end_idx]
        tar = trainenc.input_ids[:, start_idx + 1:end_idx + 1]
        tar[:, -1] = -100  # Ignore the last token in target
        trainloader.append((inp, tar))

    # Generate validation data samples
    validloader = []
    for _ in range(nsamples):
        while True:
            # Randomly select a sample from the test data
            i = random.randint(0, len(testdata) - 1)
            # Tokenize the selected text
            valenc = tokenizer(testdata[i]['text'], return_tensors='pt')

            # Ensure the sequence length is larger than `seqlen`
            if valenc.input_ids.shape[1] > seqlen:
                break  # Continue if valid sequence length is found

        # Randomly select a window from the tokenized text
        start_idx = random.randint(0, valenc.input_ids.shape[1] - seqlen - 1)
        end_idx = start_idx + seqlen
        inp = valenc.input_ids[:, start_idx:end_idx]
        tar = valenc.input_ids[:, start_idx + 1:end_idx + 1]
        tar[:, -1] = -100  # Ignore the last token in target
        validloader.append((inp, tar))

    return trainloader, validloader


# Load and process c4 dataset
def get_c4(nsamples, seed, seqlen, tokenizer):
    # Load train and validation datasets
    # traindata = load_dataset('allenai/c4', 'allenai--c4', data_files={'train': 'en/c4-train.00000-of-01024.json.gz'}, split='train')
    # valdata = load_dataset('allenai/c4', 'allenai--c4', data_files={'validation': 'en/c4-validation.00000-of-00008.json.gz'}, split='validation')

    # dataset = load_dataset('json', data_files={'train': 'https://huggingface.co/datasets/allenai/c4/resolve/main/en/c4-train.00000-of-01024.json.gz', 'validation': 'https://huggingface.co/datasets/allenai/c4/resolve/main/en/c4-validation.00000-of-00008.json.gz'})
    # traindata = dataset['train']
    # valdata = dataset['validation']
    traindata = load_from_disk("datasets/c4/train")
    valdata = load_from_disk("datasets/c4/validation")

    # Generate samples from training set
    random.seed(seed)
    trainloader = []
    for _ in range(nsamples):
        while True:
            i = random.randint(0, len(traindata) - 1)
            trainenc = tokenizer(traindata[i]['text'], return_tensors='pt')
            if trainenc.input_ids.shape[1] > seqlen:
                break
        i = random.randint(0, trainenc.input_ids.shape[1] - seqlen - 1)
        j = i + seqlen
        inp = trainenc.input_ids[:, i:j]
        tar = trainenc.input_ids[:, i + 1:j + 1] 
        tar[:, -1] = -100
        trainloader.append((inp, tar))

    # Prepare validation dataset
    validloader = []
    for _ in range(nsamples):
        while True:
            i = random.randint(0, len(valdata) - 1)
            valenc = tokenizer(valdata[i]['text'], return_tensors='pt')
            if valenc.input_ids.shape[1] > seqlen:
                break
        i = random.randint(0, valenc.input_ids.shape[1] - seqlen - 1)
        j = i + seqlen
        inp = valenc.input_ids[:, i:j]
        tar = valenc.input_ids[:, i + 1:j + 1] 
        tar[:, -1] = -100
        validloader.append((inp, tar))

    return trainloader, validloader


def get_medicine(nsamples, seed, seqlen, tokenizer):
    # Load train and test datasets
    dataset = load_dataset("AdaptLLM/med_knowledge_prob", "Medicine")

    data = dataset['test'].shuffle(seed=42).select(range(2 * nsamples))['exp']
    # Remove None values
    data = [d for d in data if d is not None]
    train = data[:nsamples]
    test = data[nsamples:]

    # Initialize random seed
    random.seed(seed)

    # Generate training data samples
    trainloader = []
    for _ in range(nsamples):
        while True:
            # Randomly select a sample from the training data
            i = random.randint(0, len(train) - 1)
            # Tokenize the selected text
            trainenc = tokenizer(train[i], return_tensors='pt')

            # Ensure the sequence length is larger than `seqlen`
            if trainenc.input_ids.shape[1] > seqlen:
                break  # Continue if valid sequence length is found

        # Randomly select a window from the tokenized text
        start_idx = random.randint(0, trainenc.input_ids.shape[1] - seqlen - 1)
        end_idx = start_idx + seqlen
        inp = trainenc.input_ids[:, start_idx:end_idx]
        tar = trainenc.input_ids[:, start_idx + 1:end_idx + 1]
        tar[:, -1] = -100  # Ignore the last token in target
        trainloader.append((inp, tar))

    # Generate test data samples
    testloader = []
    for _ in range(nsamples):
        while True:
            # Randomly select a sample from the test data
            i = random.randint(0, len(test) - 1)
            # Tokenize the selected text
            testenc = tokenizer(test[i], return_tensors='pt')

            # Ensure the sequence length is larger than `seqlen`
            if testenc.input_ids.shape[1] > seqlen:
                break  # Continue if valid sequence length is found

        # Randomly select a window from the tokenized text
        start_idx = random.randint(0, testenc.input_ids.shape[1] - seqlen - 1)
        end_idx = start_idx + seqlen
        inp = testenc.input_ids[:, start_idx:end_idx]
        tar = testenc.input_ids[:, start_idx + 1:end_idx + 1]
        tar[:, -1] = -100  # Ignore the last token in target
        testloader.append((inp, tar))

    return trainloader, testloader



def get_law(nsamples, seed, seqlen, tokenizer):
    # Load train and test datasets
    dataset = load_dataset("AdaptLLM/law_knowledge_prob")

    data = dataset['test'].shuffle(seed=42).select(range(2 * nsamples))['text']
    train = data[:nsamples]
    test = data[nsamples:]

    # Initialize random seed
    random.seed(seed)

    # Generate training data samples
    trainloader = []
    for _ in range(nsamples):
        while True:
            # Randomly select a sample from the training data
            i = random.randint(0, len(train) - 1)
            # Tokenize the selected text (no longer join entire dataset)
            trainenc = tokenizer(train[i], return_tensors='pt')

            # Ensure the sequence length is larger than `seqlen`
            if trainenc.input_ids.shape[1] > seqlen:
                break  # Continue if valid sequence length is found

        # Randomly select a window from the tokenized text
        start_idx = random.randint(0, trainenc.input_ids.shape[1] - seqlen - 1)
        end_idx = start_idx + seqlen
        inp = trainenc.input_ids[:, start_idx:end_idx]
        tar = trainenc.input_ids[:, start_idx + 1:end_idx + 1]
        tar[:, -1] = -100  # Ignore the last token in target
        trainloader.append((inp, tar))

    # Generate test data samples
    testloader = []
    for _ in range(nsamples):
        while True:
            # Randomly select a sample from the test data
            i = random.randint(0, len(test) - 1)
            # Tokenize the selected text
            testenc = tokenizer(test[i], return_tensors='pt')

            # Ensure the sequence length is larger than `seqlen`
            if testenc.input_ids.shape[1] > seqlen:
                break  # Continue if valid sequence length is found

        # Randomly select a window from the tokenized text
        start_idx = random.randint(0, testenc.input_ids.shape[1] - seqlen - 1)
        end_idx = start_idx + seqlen
        inp = testenc.input_ids[:, start_idx:end_idx]
        tar = testenc.input_ids[:, start_idx + 1:end_idx + 1]
        tar[:, -1] = -100  # Ignore the last token in target
        testloader.append((inp, tar))

    return trainloader, testloader


# Load and process finance dataset
def get_finance(nsamples, seed, seqlen, tokenizer):
    # Load train and test datasets
    dataset = load_dataset("ZixuanKe/finance-unsup")

    data = dataset['train'].shuffle(seed=42).select(range(2 * nsamples))['text']
    train = data[:nsamples]
    test = data[nsamples:]

    # Encode datasets
    trainenc = tokenizer(" ".join(train), return_tensors='pt')
    testenc = tokenizer(" ".join(test), return_tensors='pt')

    # Generate samples from training set
    random.seed(seed)
    trainloader = []
    for _ in range(nsamples):
        i = random.randint(0, trainenc.input_ids.shape[1] - seqlen - 1)
        j = i + seqlen
        inp = trainenc.input_ids[:, i:j]
        tar = trainenc.input_ids[:, i + 1:j + 1] 
        tar[:, -1] = -100

        trainloader.append((inp, tar))

    testloader = []
    for _ in range(nsamples):
        i = random.randint(0, testenc.input_ids.shape[1] - seqlen - 1)
        j = i + seqlen
        inp = testenc.input_ids[:, i:j]
        tar = testenc.input_ids[:, i + 1:j + 1] 
        tar[:, -1] = -100
        testloader.append((inp, tar))

    return trainloader, testloader

# Function to select the appropriate loader based on dataset name
def get_loaders(name, nsamples=128, seed=0, seqlen=2048, tokenizer=None):
    if 'wikitext2' in name:
        return get_wikitext2(nsamples, seed, seqlen, tokenizer)
    if "c4" in name:
        return get_c4(nsamples, seed, seqlen, tokenizer)
    if "finance" in name:
        return get_finance(nsamples, seed, seqlen, tokenizer)
    if "law" in name:
        return get_law(nsamples, seed, seqlen, tokenizer)
    if "medicine" in name:
        return get_medicine(nsamples, seed, seqlen, tokenizer)