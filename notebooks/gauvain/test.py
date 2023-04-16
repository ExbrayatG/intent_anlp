import torch
from datasets import load_dataset
from tqdm import tqdm
from nltk.tokenize import TweetTokenizer
import pandas as pd
from termcolor import colored
from collections import Counter
from torchtext.vocab import vocab, FastText
from torch.utils.data import Dataset
import numpy as np
from torch.utils.data import DataLoader
from torchinfo import summary
import torch.optim as optim
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sn

# Loading datasets
silicone_swda = load_dataset("silicone", "swda")
silicone_mrda = load_dataset("silicone", "mrda")

# Printing datasets metrics
print("MRDA train labels: ", Counter(silicone_mrda["train"]["Label"]))
print("MRDA test labels: ", Counter(silicone_mrda["test"]["Label"]))
print("MRDA sample: ", silicone_mrda["train"][5])
print(
    "MRDA max utterance length",
    max([len(ut) for ut in silicone_mrda["train"]["Utterance"]]),
)
print(
    "MRDA average utterance length: ",
    sum([len(ut) for ut in silicone_mrda["train"]["Utterance"]])
    / len([len(ut) for ut in silicone_mrda["train"]["Utterance"]]),
)

# Load pretrained vectors
pretrained_vectors = FastText(language="simple")

# Preprocessing of the embeddings
pretrained_vocab = vocab(pretrained_vectors.stoi)
# Insert unknown token
unk_token = "<unk>"
unk_index = 0
pretrained_vocab.insert_token("<unk>", unk_index)
# Insert padding token
pad_token = "<pad>"
pad_index = 1
pretrained_vocab.insert_token("<pad>", pad_index)

# this is necessary otherwise it will throw runtime error if OOV token is queried
pretrained_vocab.set_default_index(unk_index)
pretrained_embeddings = pretrained_vectors.vectors
pretrained_embeddings = torch.cat(
    (torch.zeros(1, pretrained_embeddings.shape[1]), pretrained_embeddings)
)
pretrained_embeddings.size()

# Tokenizing
tok = TweetTokenizer()


def tokenize_pad_numericalize(entry, vocab_stoi, max_length=20):
    text = [
        vocab_stoi[token] if token in vocab_stoi else vocab_stoi["<unk>"]
        for token in tok.tokenize(entry.lower())
    ]
    padded_text = None
    if len(text) < max_length:
        padded_text = text + [vocab_stoi["<pad>"] for i in range(len(text), max_length)]
    elif len(text) > max_length:
        padded_text = text[:max_length]
    else:
        padded_text = text
    return padded_text


def tokenize_all(entries, vocab_stoi):
    res = {}
    res["Utterance"] = [
        tokenize_pad_numericalize(entry, vocab_stoi, max_length=50)
        for entry in entries["Utterance"]
    ]
    res["Label"] = entries["Label"]
    return res


silicone_mrda["train"] = silicone_mrda["train"].map(
    lambda e: tokenize_all(e, pretrained_vocab.get_stoi()), batched=True
)
print("Tokenized utterance: ", silicone_mrda["train"]["Utterance"][2])

silicone_mrda["validation"] = silicone_mrda["validation"].map(
    lambda e: tokenize_all(e, pretrained_vocab.get_stoi()), batched=True
)
silicone_mrda["test"] = silicone_mrda["test"].map(
    lambda e: tokenize_all(e, pretrained_vocab.get_stoi()), batched=True
)


# Data loaders
class UtteranceDataset(Dataset):
    def __init__(self, data, args):
        self.args = args
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = {
            "Utterance": np.array(self.data[idx]["Utterance"]),
            "Label": np.array(self.data[idx]["Label"]),
        }
        return item


# Create DataLoader
args = {"bsize": 64}

train_loader = DataLoader(
    UtteranceDataset(silicone_mrda["train"], args),
    batch_size=args["bsize"],
    num_workers=2,
    shuffle=True,
    drop_last=True,
)
val_loader = DataLoader(
    UtteranceDataset(silicone_mrda["validation"], args),
    batch_size=args["bsize"],
    num_workers=2,
    shuffle=True,
    drop_last=True,
)
test_loader = DataLoader(
    UtteranceDataset(silicone_mrda["test"], args),
    batch_size=args["bsize"],
    num_workers=2,
    shuffle=True,
    drop_last=True,
)


# Creating the model
class BasicMLP(torch.nn.Module):
    def __init__(self, D_in, H, D_out, pretrained_vectors=None):
        """
        In the constructor we instantiate two nn.Linear modules and assign them as
        member variables.
        """
        super(BasicMLP, self).__init__()

        self.ebd = torch.nn.Embedding.from_pretrained(pretrained_vectors, freeze=True)
        self.hidden_layer = torch.nn.Linear(H, H, bias=True)
        self.classification_layer = torch.nn.Linear(H, D_out, bias=True)
        self.softmax = torch.nn.Softmax(dim=1)

        # define the dropout strategy (here, 20% (0.2) of the vector is ignored to prevent overfitting)
        # we don't use it here but it's a good thing to keep in mind
        # self.dropout = nn.Dropout(p=0.2)

    def forward(self, x):
        """
        In the forward function we accept a Tensor of input data and we must return
        a Tensor of output data. We can use Modules defined in the constructor as
        well as arbitrary operators on Tensors.
        """
        x = self.ebd(x)
        x = x.mean(1)  # Pourquoi?
        h = torch.relu(self.hidden_layer(x))
        # h = self.dropout(h)
        h = self.classification_layer(h)
        logits = self.softmax(h)

        return logits


sizes = next(iter(train_loader))["Utterance"].size()
batchsize = sizes[0]
inputdim = sizes[1]
print("batchsize, inputdim: ", batchsize, inputdim)

hiddendim = 300  # dimension of the pretrained vector
outputdim = len(set(silicone_mrda["train"]["Label"]))
print("hiddendim, outputdim: ", hiddendim, outputdim)

# Instantiate the model
utterance_model = BasicMLP(
    inputdim, hiddendim, outputdim, pretrained_vectors=pretrained_vectors.vectors
)
print("model: ", utterance_model)

if torch.cuda.is_available():
    device = "cuda"
    print("DEVICE = ", colored(torch.cuda.get_device_name(0), "green"))
else:
    device = "cpu"
    print("DEVICE = ", colored("CPU", "blue"))
utterance_model.to(device)

print(
    summary(
        utterance_model.to("cpu"),
        (batchsize, inputdim),
        dtypes=["torch.IntTensor"],
        device="cpu",
        verbose=2,
    )
)
utterance_model.to(device)


def train(model, optimizer, ep, args):
    # set the model into a training mode
    model.train()

    # Empty lists for loss and accuracy
    loss_it = list()
    acc_it = list()

    # start the loop over all the training batches
    for it, batch in tqdm(
        enumerate(train_loader), desc="Epoch %s:" % (ep), total=train_loader.__len__()
    ):
        batch = {
            "Utterance": batch["Utterance"].to("cpu"),
            "Label": batch["Label"].to("cpu"),
        }

        # Reset optimizer
        optimizer.zero_grad()

        # apply the model on the batch
        logits = model(batch["Utterance"])

        # Computing weights according to inverse frequency of labels
        b_counter = Counter(batch["Label"].detach().cpu().tolist())
        b_weights = torch.tensor(
            [
                sum(batch["Label"].detach().cpu().tolist()) / b_counter[label]
                if b_counter[label] > 0
                else 0
                for label in list(range(args["num_class"]))
            ]
        )
        b_weights = b_weights.to("cpu")

        # Loss
        loss_function = torch.nn.CrossEntropyLoss(weight=b_weights)
        loss = loss_function(logits, batch["Label"])

        # Computing backpropagation
        loss.backward()

        # indicate to the optimizer we've done a step
        optimizer.step()

        # Adding loss value to current iteration list
        loss_it.append(loss.item())

        # Get predicted class
        _, tag_seq = torch.max(logits, 1)

        # Computing accuracy
        correct = (tag_seq.flatten() == batch["Label"].flatten()).float().sum()
        acc = correct / batch["Label"].flatten().size(0)
        acc_it.append(acc.item())

    # Losses and accuracies average for the epoch
    loss_it_avg = sum(loss_it) / len(loss_it)
    acc_it_avg = sum(acc_it) / len(acc_it)

    print(
        "Epoch %s/%s : %s : (%s %s) (%s %s)"
        % (
            colored(str(ep), "blue"),
            args["max_eps"],
            colored("Training", "blue"),
            colored("loss", "cyan"),
            sum(loss_it) / len(loss_it),
            colored("acc", "cyan"),
            sum(acc_it) / len(acc_it),
        )
    )


def inference(target, loader, model):
    # set model into evaluation mode
    model.eval()

    # Empty lists for loss, accuracy, f1-score, predicted and true values
    loss_it = list()
    acc_it = list()
    f1_it = list()
    preds = list()
    trues = list()

    for it, batch in tqdm(
        enumerate(loader), desc="%s:" % (target), total=loader.__len__()
    ):
        with torch.no_grad():
            batch = {
                "Utterance": batch["Utterance"].to("cpu"),
                "Label": batch["Label"].to("cpu"),
            }

            # apply the model
            logits = model(batch["Utterance"])

            # Loss
            loss_function = torch.nn.CrossEntropyLoss()
            loss = loss_function(logits, batch["Label"])
            loss_it.append(loss.item())

            # Get predicted class
            _, tag_seq = torch.max(logits, 1)

            # Accuracy
            correct = (tag_seq.flatten() == batch["Label"].flatten()).float().sum()
            acc = correct / batch["Label"].flatten().size(0)
            acc_it.append(acc.item())

            # Preds and trues
            preds.extend(tag_seq.cpu().detach().tolist())
            trues.extend(batch["Label"].cpu().detach().tolist())

    # compute the average loss and accuracy accross the iterations (batches)
    loss_it_avg = sum(loss_it) / len(loss_it)
    acc_it_avg = sum(acc_it) / len(acc_it)

    # print useful information. Important during training as we want to know the performance over the validation set after each epoch
    print(
        "%s : (%s %s) (%s %s)"
        % (
            colored(target, "blue"),
            colored("loss", "cyan"),
            sum(loss_it) / len(loss_it),
            colored("acc", "cyan"),
            sum(acc_it) / len(acc_it),
        )
    )

    # return the true and predicted values with the losses and accuracies
    return trues, preds, loss_it_avg, acc_it_avg, loss_it, acc_it


def run_epochs(model, args):
    # Adam optimizer
    optimizer = optim.Adam(model.parameters(), lr=args["lr"])

    # Validation losses
    val_ep_losses = list()

    for ep in range(args["max_eps"]):
        train(model, optimizer, ep, args)
        (
            trues,
            preds,
            val_loss_it_avg,
            val_acc_it_avg,
            val_loss_it,
            val_acc_it,
        ) = inference("validation", val_loader, model)

        # append the validation losses
        val_ep_losses.append(val_loss_it_avg)

    return val_ep_losses


# here you can specify if you want a GPU or a CPU by setting the cuda argument as -1 for CPU and another index for GPU. If you only have one GPU, put 0.
args.update({"max_eps": 10, "lr": 0.001, "num_class": outputdim})

# Instantiate model
# model = TweetModel(pretrained_embeddings, args['num_class'], args, dimension=50, freeze_embeddings = True )
utterance_model = BasicMLP(
    inputdim, hiddendim, outputdim, pretrained_vectors=pretrained_vectors.vectors
)
loss_list_val = run_epochs(utterance_model, args)


def plot_loss(loss_list):
    """
    this function creates a plot. a simple curve showing the different values at each steps.
    Here we use it to plot the loss so we named it plot_loss, but the same function with different titles could be used to plot accuracies
    or other metrics for instance.

    Args:
      loss_list (list of floats): list of numerical values
    """
    plt.plot(range(len(loss_list)), loss_list)
    plt.xlabel("epochs")
    # in our model we use Softmax then NLLLoss which means Cross Entropy loss
    plt.ylabel("Cross Entropy")
    # in our training loop we used an Adam optimizer so we indicate it there
    plt.title("lr: {}, optim_alg:{}".format(args["lr"], "Adam"))
    # let's directly show the plot when calling this function
    plt.show()


# plot_loss(loss_list_val)

trues, preds, loss_it_avg, acc_it_avg, loss_it, acc_it = inference(
    "test", test_loader, utterance_model
)

# let's look at the first ten predictions
for t, p in zip(trues[:30], preds[:30]):
    correct = colored("Correct", "green") if t == p else colored("Mistake", "red")
    print("true", t, "predicted", p, correct)

print(classification_report(np.array(trues).flatten(), np.array(preds).flatten()))

cm = confusion_matrix(np.array(trues).flatten(), np.array(preds).flatten())
df_cm = pd.DataFrame(cm)
# config plot sizes
sn.set(font_scale=1)
sn.heatmap(
    df_cm, annot=True, annot_kws={"size": 8}, cmap="coolwarm", linewidth=0.5, fmt=""
)
plt.show()
