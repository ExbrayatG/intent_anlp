import torch
from torchtext.vocab import FastText
from torch.utils.data import Dataset, DataLoader
from src.data import create_dataloaders, DialogueDataset


class CNN(torch.nn.Module):
    def __init__(self, dataset, D_in, H, D_out, pretrained_vectors=None):

        super(CNN, self).__init__()

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
        self.conv = torch.nn.Conv1d(H, H, kernel_size=3)
        self.pool = torch.nn.AdaptiveMaxPool1d(1)
        self.dropout = torch.nn.Dropout(0.5)

        self.fc1 = torch.nn.Linear(H, H, bias = True)
        self.fc2 = torch.nn.Linear(H, H, bias = True)

        self.classification_layer = torch.nn.Linear(H, D_out, bias = True)
        self.softmax = torch.nn.Softmax(dim=1)

    def forward(self, x):
        
        # Compute the embeddings
        x = self.ebd(x)

        # Classify
        x = x.transpose(1, 2)  # Appliquer une transformation avant la couche de convolution

        x = self.conv(x)
        x = self.pool(x).squeeze()
        x = x.view(x.size(0), -1)

        x = self.dropout(x)

        x = self.fc1(x)
        x = self.fc2(x)

        x = self.dropout(x)
    
        h = self.classification_layer(x)

        # Return logit
        logits = self.softmax(h)
        
        return logits