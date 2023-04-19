import torch
from torchtext.vocab import FastText
from torch.utils.data import Dataset, DataLoader
from src.data import create_dataloaders, DialogueDataset


class MLP(torch.nn.Module):
    def __init__(self, dataset, D_in, H, D_out, pretrained_vectors=None):

        super(MLP, self).__init__()

        self.dataset = dataset
        self.train_dataloder = None
        self.val_dataloader = None
        
        # Tokenize
        def encode(self, dataset):
            def tokenize(sentence):
                return sentence.split()

            def get_avg_embedding(text):
                embeddings = [self.fasttext_embedding[word] for word in tokenize(text)]
                return sum(embeddings) / len(embeddings)

            return [get_avg_embedding(text) for text in dataset]
        
        tokenized_dataset = self.encode(dataset)

        # Put in dataloader
        self.train_dataloder, _ , self.val_dataloader = create_dataloaders(tokenized_dataset["train"]["Utterance"], 
                                                                           tokenized_dataset["validation"]["Utterance"],
                                                                           tokenized_dataset["test"]["Utterance"],
                                                                           tokenized_dataset, batch_size=16)
        

        # Create the embedding model
        self.ebd = torch.nn.Embedding.from_pretrained(pretrained_vectors, freeze=True)


        # Create the classification model
        self.hidden_layer1 = torch.nn.Linear(4*H, 4*H, bias = True)
        self.hidden_layer2 = torch.nn.Linear(4*H, 4*H, bias = True)
        self.hidden_layer3 = torch.nn.Linear(4*H, H, bias = True)

        self.classification_layer = torch.nn.Linear(H, D_out, bias = True)
        self.softmax = torch.nn.Softmax(dim=1)

        self.dropout = torch.nn.Dropout(p=0.5)

    def forward(self, x):
        
        # Compute the embedding
        x = self.ebd(x)
        x  = x.mean(1) #Averaging embeddings

        # Classify
        x = torch.relu(self.hidden_layer1(x))
        x = torch.relu(self.hidden_layer2(x))
        x = torch.relu(self.hidden_layer3(x))

        x = self.dropout(x)
    
        h = self.classification_layer(x)

        # Return logit
        logits = self.softmax(h)
        
        return logits