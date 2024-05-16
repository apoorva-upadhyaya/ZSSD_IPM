import torch
import torch.nn as nn
import torch.optim as optim
from transformers import AutoTokenizer, AutoModel
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import accuracy_score,f1_score, precision_score, recall_score
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
import csv
import pandas as pd
import ast
import json


class CreateDataset(Dataset):
    def __init__(self, texts,rationals,targets, comet_intents,llm_intents,stances,tweet_tokenizer,tokenizer, max_length):
        self.texts = texts
        self.targets = targets
        self.stances = stances
        self.llm_intents=llm_intents
        self.comet_intents=comet_intents
        self.rationals = rationals
        self.tweet_tokenizer = tweet_tokenizer
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        target = self.targets[idx]
        llm_intent = self.llm_intents[idx]
        comet_intent = self.comet_intents[idx]
        rational=self.rationals[idx]
        stance = self.stances[idx]

        # Tokenize input text,target,comet_intent,llm_intent
        text_target=text+target
        inputs = self.tweet_tokenizer(text_target, return_tensors="pt", max_length=self.max_length, truncation=True, padding="max_length")
        text_ids = inputs["input_ids"]
        text_mask = inputs["attention_mask"]

        inputs = self.tokenizer(comet_intent, return_tensors="pt", max_length=self.max_length, truncation=True, padding="max_length")
        comet_intent_ids = inputs["input_ids"]
        comet_intent_mask = inputs["attention_mask"]

        inputs = self.tokenizer(llm_intent, return_tensors="pt", max_length=self.max_length, truncation=True, padding="max_length")
        llm_intent_ids = inputs["input_ids"]
        llm_intent_mask = inputs["attention_mask"]

        # Tokenize target rational output
        text_tokens = self.tweet_tokenizer.tokenize(text)
        rational_tokens = self.tweet_tokenizer.tokenize(rational)
        rational_binary = [1 if token in rational_tokens else 0 for token in text_tokens]
        # Pad the binary values
        rational_binary += [0] * (self.max_length - len(rational_binary))

        return text_ids, text_mask,comet_intent_ids,comet_intent_mask, llm_intent_ids,llm_intent_mask,torch.tensor(rational_binary), torch.tensor(stance)


class RationalModel(nn.Module):
    def __init__(self, bertweet_model,bert_model, hidden_dim, rank,num_heads,hidden_size,dropout,output_size):
        super(RationalModel, self).__init__()
        self.bertweet = bertweet_model
        self.bert = bert_model
        self.rational_head = FactorizedBilinearPooling(hidden_dim, rank,rank,hidden_size)
        # Multi-Head Attention
        self.multihead_attn = nn.MultiheadAttention(hidden_size, num_heads)   
        self.norm = nn.LayerNorm(hidden_size, num_heads)
        self.dropout = nn.Dropout(dropout) 
        # Final Linear layer
        self.fc1 = nn.Linear(2*hidden_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_size, output_size)

    def forward(self, text_ids, text_mask, comet_intent_ids, comet_intent_mask, llm_intent_ids, llm_intent_mask):

        text_ids = text_ids.squeeze(1)
        text_mask = text_mask.squeeze(1)
        llm_intent_ids = llm_intent_ids.squeeze(1)
        llm_intent_mask = llm_intent_mask.squeeze(1)
        comet_intent_ids = comet_intent_ids.squeeze(1)
        comet_intent_mask = comet_intent_mask.squeeze(1)

        text_output = self.bertweet(input_ids=text_ids, attention_mask=text_mask)[0][:, 0, :]  # [CLS] token
        
        comet_intent_output = self.bert(input_ids=comet_intent_ids, attention_mask=comet_intent_mask)[0][:, 0, :]  # [CLS] token
        llm_intent_output = self.bert(input_ids=llm_intent_ids, attention_mask=llm_intent_mask)[0][:, 0, :]  # [CLS] token
        # Concat comet intent and LLM intent
        concat_intent=torch.cat((comet_intent_output, llm_intent_output),dim=-1)
        concat_intent = self.fc1(concat_intent)
        print("concat_intent::",concat_intent.size())
        print("text_output::",text_output.size())
        
        # Factorized bilinear pooling
        rational_head_output = self.rational_head(text_output, concat_intent)
        print("rational_head_output::",rational_head_output.size())
        # Multi-Head Attention
        attn_output, _ = self.multihead_attn(rational_head_output, rational_head_output, rational_head_output)
        print("attn_output::",attn_output.size())

        # Layer normalization
        pooling_output = self.norm(rational_head_output + self.dropout(attn_output))
        print("pooling_output::",pooling_output.size())
        
        rational_logits = self.fc2(pooling_output)
        return rational_logits.squeeze(1)

class FactorizedBilinearPooling(nn.Module):
    def __init__(self, input_size, rank,window_size,output_size):
        super(FactorizedBilinearPooling, self).__init__()
        self.U = nn.Linear(input_size, rank, bias=False)
        self.V = nn.Linear(input_size, rank, bias=False)
        self.window_size = window_size
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x1, x2):
        x1 = self.U(x1)
        x2 = self.V(x2)   
        # Element-wise multiplication
        interaction = x1 * x2
        batch_size, rank = interaction.size()
        interaction = interaction.view(batch_size, rank // self.window_size, self.window_size)
        # Sum over all non-overlapping windows of size h
        pooled_interaction = torch.sum(interaction, dim=-2)
        # Flatten the tensor
        pooled_interaction = pooled_interaction.view(batch_size, -1)
        # L2 normalization
        pooled_interaction = F.normalize(pooled_interaction, p=2, dim=-1)
        pooled_interaction=self.fc(pooled_interaction)
        return pooled_interaction



class StanceModel_Stage1(nn.Module):
    def __init__(self,hidden_size,num_classes):
        super(StanceModel_Stage1, self).__init__()       
        # Linear layer
        self.fc1 = nn.Linear(hidden_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, num_classes)

    def forward(self, rational):
        fc_output = self.fc1(rational)
        stance_output = self.fc2(fc_output)
        return stance_output

def reconstructWords(tokens):
    words = []
    current_word = ""
    for token in tokens:
        if "@@" in token:
            # Remove '@@' symbol and concatenate with current word
            current_word += token.replace("@@", "")
        else:
            # If no '@@' symbol, it's a separate word
            if current_word:
                # If current_word is not empty, add it to words list
                words.append(current_word)
                current_word = ""  # Reset current_word
            words.append(token)  # Add token to words list

    # Add the last word if current_word is not empty
    if current_word:
        words.append(current_word)
    return words

def backtrack_rational_output(input_ids, rational_logits, tokenizer):
    rational_output_tokens = []
    for i in range(len(input_ids)):
        input_tokens = tokenizer.convert_ids_to_tokens(input_ids[i], skip_special_tokens=True)
        rational_binary = rational_logits[i].sigmoid().round().cpu().numpy()  # Round to convert logits to binary values
        rational_tokens = [token for token, is_rational in zip(input_tokens, rational_binary) if is_rational == 1]
        rational_tokens_construct=reconstructWords(rational_tokens)
        rational_output_tokens.append(rational_tokens_construct)
    return rational_output_tokens


# Define stance label mapping
label_mapping = {"none": 0, "favor": 1, "against": 2}

data=pd.read_csv("../data/rational_train_clean.csv", delimiter=";") 
print("train data :: ",len(data))

# Train targets
li_train=["atheism","climate Change is a real concern","feminist movement","hillary clinton"]

# Test target
li_test=["legalization of abortion"]

train_texts=[]
train_rationals=[]
train_comet_intents=[]
train_llm_intents=[]
train_stances=[]
train_targets=[]

for i in range(len(data)):
    id_=str(data.tweetid.values[i])
    target=(str(data.target.values[i]))
    if target in li_train:
        train_texts.append(str(data.text.values[i]))
        train_rationals.append(str(data.rational.values[i]))
        train_stances.append(str(data.stance.values[i]))
        train_targets.append(str(data.target.values[i]))
        train_comet_intents.append(str(data.comet_intent.values[i]))
        train_llm_intents.append(str(data.llm_intent.values[i]))


train_stances = [label_mapping[label] for label in train_stances]

test_texts=[]
test_rationals=[]
test_stances=[]
test_targets=[]
test_comet_intents=[]
test_llm_intents=[]

data=pd.read_csv("../data/rational_test_clean.csv", delimiter=";") 
print("test data :: ",len(data))

for i in range(len(data)):
    id_=str(data.tweetid.values[i])
    target=(str(data.target.values[i]))
    if target in li_test:
        test_texts.append(str(data.text.values[i]))
        test_rationals.append(str(data.rational.values[i]))
        test_stances.append(str(data.stance.values[i]))
        test_targets.append(str(data.target.values[i]))
        test_comet_intents.append(str(data.comet_intent.values[i]))
        test_llm_intents.append(str(data.llm_intent.values[i]))

test_stances = [label_mapping[label] for label in test_stances]

#Divide into train and validation tests
train_texts, val_texts, train_rationals, val_rationals,train_targets,val_targets, train_comet_intents, val_comet_intents,train_llm_intents,val_llm_intents, train_stances, val_stances = train_test_split(train_texts, train_rationals,train_targets, train_comet_intents, train_llm_intents,train_stances, test_size=0.15, random_state=42)

# BERT tokenizer
tweet_tokenizer = AutoTokenizer.from_pretrained("vinai/bertweet-base")
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

# Define datasets and data loaders
train_dataset = CreateDataset(train_texts, train_rationals, train_targets, train_comet_intents, train_llm_intents,train_stances,tweet_tokenizer ,tokenizer, max_length=128)
val_dataset = CreateDataset(val_texts, val_rationals, val_targets, val_comet_intents, val_llm_intents,val_stances,tweet_tokenizer, tokenizer, max_length=128)
test_dataset = CreateDataset(test_texts, test_rationals, test_targets, test_comet_intents, test_llm_intents, test_stances,tweet_tokenizer, tokenizer, max_length=128)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# Model parameters
hidden_dim_rational = 768
rank = 128  # Rank for factorized bilinear pooling
num_epochs = 1
learning_rate = 0.001
num_heads=8
hidden_size=128
output_size=128
bertweet_model = AutoModel.from_pretrained("vinai/bertweet-base")
bert_model = AutoModel.from_pretrained("bert-base-uncased")
num_classes=3
dropout=0.5
# Initialize model
rational_model = RationalModel(bertweet_model,bert_model, hidden_dim_rational,rank,num_heads,hidden_size,dropout,output_size)
stance_model = StanceModel_Stage1(hidden_size,num_classes)

# Loss function and optimizer
rational_criterion = nn.BCEWithLogitsLoss()
stance_criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(list(rational_model.parameters()) + list(stance_model.parameters()), lr=learning_rate)

# Training loop
for epoch in range(num_epochs):
    print("Epoch:::::::::",epoch)
    rational_model.train()
    stance_model.train()
    total_loss = 0.0
    for text_ids, text_mask, comet_intent_ids, comet_intent_mask, llm_intent_ids, llm_intent_mask, rational_binary,stance in train_loader:
        # Forward pass for Rational task
        optimizer.zero_grad()
        rational_logits = rational_model(text_ids, text_mask, comet_intent_ids, comet_intent_mask, llm_intent_ids, llm_intent_mask)
        # Calculate loss for Rational task
        rational_loss = rational_criterion(rational_logits, rational_binary.float())
        # Forward pass for Stance task
        stance_logits = stance_model(rational_logits)
        # Calculate loss for Stance task
        stance_loss = stance_criterion(stance_logits, stance)
        
        # Combine rational and stance losses
        loss = rational_loss + stance_loss
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()

    # Validation loop
    rational_model.eval()
    stance_model.eval()
    val_losses = []
    val_predictions = []
    val_targets = []
    with torch.no_grad():
        for text_ids, text_mask, comet_intent_ids, comet_intent_mask, llm_intent_ids, llm_intent_mask, rational_binary,stance in val_loader:
            rational_logits = rational_model(text_ids, text_mask, comet_intent_ids, comet_intent_mask, llm_intent_ids, llm_intent_mask)
            stance_logits = stance_model(rational_logits)
            # Calculate loss
            rational_loss = rational_criterion(rational_logits, rational_binary.float())
            stance_loss = stance_criterion(stance_logits, stance)
            val_loss = rational_loss + stance_loss
            val_losses.append(val_loss.item())
            
    avg_val_loss = sum(val_losses) / len(val_losses)
    print(f"Epoch {epoch+1}/{num_epochs}, Validation Loss: {avg_val_loss}")


# Save the trained model
torch.save(rational_model.state_dict(), "models/RationalModel.pth")
torch.save(stance_model.state_dict(), "models/StanceModel_Stage1.pth")

# Load the saved model for testing
loaded_rational_model = RationalModel(bertweet_model,bert_model, hidden_dim_rational, rank,num_heads,hidden_size,dropout,output_size)
loaded_rational_model.load_state_dict(torch.load("models/RationalModel.pth"))


# Testing the rational model
loaded_rational_model.eval()
test_losses = []
test_predictions = []
test_targets = []
test_probabilities = []

with torch.no_grad():
    for text_ids, text_mask, comet_intent_ids, comet_intent_mask, llm_intent_ids, llm_intent_mask, rational_binary,stance in test_loader:

        rational_logits = rational_model(text_ids, text_mask, comet_intent_ids, comet_intent_mask, llm_intent_ids, llm_intent_mask)
        text_ids = text_ids.squeeze(1)
        rational_output_tokens = backtrack_rational_output(text_ids, rational_logits, tweet_tokenizer)
        print("rational_output_tokens::::",rational_output_tokens)

        # Calculate loss for Rational task
        loss = rational_criterion(rational_logits, rational_binary.float())
        test_losses.append(loss.item())
        
        # Calculate predictions
        predictions = (torch.sigmoid(rational_logits) > 0.5).int()
        test_predictions.extend(predictions.cpu().numpy())        
        # Save targets
        test_targets.extend(rational_binary.cpu().numpy())
        
        # Calculate prediction probabilities
        probabilities = torch.sigmoid(rational_logits).cpu().numpy()
        test_probabilities.extend(probabilities)


# Calculate average test loss
avg_test_loss = sum(test_losses) / len(test_losses)
print(f"Rational Average Test Loss: {avg_test_loss}")

# Save predictions, targets, and probabilities
with open("predictions/Stage1_try.csv", "w", newline="", encoding="utf-8") as csvfile:
    fieldnames = ["True", "Prediction", "Probability"]
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    writer.writeheader()

    for target, prediction, probability in zip(test_targets, test_predictions, test_probabilities):
        writer.writerow({
            "True": target,
            "Prediction": prediction,
            "Probability": probability
        })
        
# Calculate test accuracy
test_accuracy = accuracy_score(test_targets, test_predictions)
print(f"Rational Test Accuracy: {test_accuracy}")

