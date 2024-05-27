import torch
import torch.nn as nn
import torch.optim as optim
from transformers import AutoTokenizer, AutoModel
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import accuracy_score,classification_report, confusion_matrix
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
import csv
import pandas as pd
import ast

class CreateDataset(Dataset):
    def __init__(self, ids, texts,llm1_rationals,llm2_rationals,aux_features,targets,comet_intents,llm_intents,true_stances,llm1_stances,llm2_stances,tokenizer, max_length):
        self.ids = ids
        self.texts = texts
        self.llm1_rationals = llm1_rationals
        self.llm2_rationals = llm2_rationals
        self.aux_features = aux_features
        self.targets = targets
        self.llm_intents=llm_intents
        self.comet_intents=comet_intents
        self.true_stances = true_stances
        self.llm1_stances = llm1_stances
        self.llm2_stances = llm2_stances
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        id_ = self.ids[idx]
        text = self.texts[idx]
        llm1_rational = self.llm1_rationals[idx]
        llm2_rational = self.llm2_rationals[idx]
        aux_feature = self.aux_features[idx]
        target = self.targets[idx]
        llm_intent = self.llm_intents[idx]
        comet_intent = self.comet_intents[idx]
        true_stance = self.true_stances[idx]
        llm1_stance = self.llm1_stances[idx]
        llm2_stance = self.llm2_stances[idx]

        # Tokenize input text,target,comet_intent,llm_intent
        text_target=text+target
        inputs = self.tokenizer(text_target, return_tensors="pt", max_length=self.max_length, truncation=True, padding="max_length")
        text_ids = inputs["input_ids"]
        text_mask = inputs["attention_mask"]

        inputs = self.tokenizer(comet_intent, return_tensors="pt", max_length=self.max_length, truncation=True, padding="max_length")
        comet_intent_ids = inputs["input_ids"]
        comet_intent_mask = inputs["attention_mask"]

        inputs = self.tokenizer(llm_intent, return_tensors="pt", max_length=self.max_length, truncation=True, padding="max_length")
        llm_intent_ids = inputs["input_ids"]
        llm_intent_mask = inputs["attention_mask"]

        # Tokenize llms rationals
        text_tokens = self.tokenizer.tokenize(text)
        llm1_rational_tokens = self.tokenizer.tokenize(llm1_rational)
        llm1_rational_binary = [1 if token in llm1_rational_tokens else 0 for token in text_tokens]
        # Pad the binary values
        llm1_rational_binary += [0] * (self.max_length - len(llm1_rational_binary))

        llm2_rational_tokens = self.tokenizer.tokenize(llm2_rational)
        llm2_rational_binary = [1 if token in llm2_rational_tokens else 0 for token in text_tokens]
        # Pad the binary values
        llm2_rational_binary += [0] * (self.max_length - len(llm2_rational_binary))

        return id_,text_ids,text_mask,comet_intent_ids,comet_intent_mask,llm_intent_ids,llm_intent_mask,torch.tensor(llm1_rational_binary),torch.tensor(llm2_rational_binary),aux_feature,torch.tensor(true_stance),torch.tensor(llm1_stance),torch.tensor(llm2_stance)

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

# Gated Intermediate Rational
class GatedIntermediate(nn.Module):
    def __init__(self, num_intermediate_outputs):
        super(GatedIntermediate, self).__init__()
        self.num_intermediate_outputs = num_intermediate_outputs
        self.gating_weights = nn.Parameter(torch.abs(torch.randn(num_intermediate_outputs)))

    def forward(self, intermediate_outputs):
        # Apply softmax function to gating weights to ensure they sum to 1
        gating_weights = F.softmax(self.gating_weights, dim=0)
        # Element-wise multiplication of gating weights with intermediate outputs
        gated_outputs = [gating_weights[i] * intermediate_outputs[i] for i in range(self.num_intermediate_outputs)]
        # Concatenate the gated outputs
        combined_output = torch.cat(gated_outputs, dim=1)
        return combined_output


class RationalModel(nn.Module):
    def __init__(self,bert_model, hidden_dim, rank,num_heads,hidden_size,dropout,output_size):
        super(RationalModel, self).__init__()
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

        text_output = self.bert(input_ids=text_ids, attention_mask=text_mask)[0][:, 0, :]  # [CLS] token
        
        comet_intent_output = self.bert(input_ids=comet_intent_ids, attention_mask=comet_intent_mask)[0][:, 0, :]  # [CLS] token
        llm_intent_output = self.bert(input_ids=llm_intent_ids, attention_mask=llm_intent_mask)[0][:, 0, :]  # [CLS] token
        # Concat comet intent and LLM intent
        concat_intent=torch.cat((comet_intent_output, llm_intent_output),dim=-1)
        concat_intent = self.fc1(concat_intent)
        
        # Factorized bilinear pooling
        rational_head_output = self.rational_head(text_output, concat_intent)
        
        # Multi-Head Attention
        attn_output, _ = self.multihead_attn(rational_head_output, rational_head_output, rational_head_output)
        
        # Layer normalization
        pooling_output = self.norm(rational_head_output + self.dropout(attn_output))
        
        rational_logits = self.fc2(pooling_output)
        return rational_logits.squeeze(1)

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

class StanceModel_Stage2(nn.Module):
    def __init__(self, bert_model,rank,num_heads,hidden_size,num_intermediate_outputs,dropout,num_classes):
        super(StanceModel_Stage2, self).__init__()
        self.bert = bert_model
        self.rational_head = FactorizedBilinearPooling(hidden_size, rank,rank,hidden_size)
        # Multi-Head Attention
        self.multihead_attn = nn.MultiheadAttention(hidden_size, num_heads)
        self.norm = nn.LayerNorm(hidden_size, num_heads)
        self.dropout = nn.Dropout(dropout)
        # Rational Selector
        self.gated_intermediate = GatedIntermediate(num_intermediate_outputs)
        # Final Linear layer
        self.fc1 = nn.Linear(4*hidden_size,rank)
        self.fc2 = nn.Linear(hidden_size, num_classes)

    def forward(self,text_ids,text_mask,aux_feature,loaded_rational_preds,llm1_rational_binary,llm2_rational_binary):

        text_ids = text_ids.squeeze(1)
        text_mask = text_mask.squeeze(1)
        text_output = self.bert(input_ids=text_ids, attention_mask=text_mask)[0][:, :, 0] # [CLS] token
        
        aux_features_modified = aux_feature.repeat(1, 256, 1)
        aux_features_final=torch.sum(aux_features_modified, -1)
        aux_features_final = aux_features_final.to(torch.float)
        # Factorized Bi-linear Pooling output for text and aux features
        flp_output = self.rational_head(text_output, aux_features_final)

        # Gated rational output
        gated_rational_output = self.gated_intermediate([loaded_rational_preds, llm1_rational_binary, llm2_rational_binary])

        # Concat flp(text,aux) and rational output
        concat_output=torch.cat((flp_output, gated_rational_output),dim=-1)
        concat_output = self.fc1(concat_output)

        # Multi-Head Attention
        attn_output, _ = self.multihead_attn(concat_output, concat_output, concat_output)
        # Add & Layer normalization
        attn_output1 = self.norm(concat_output + self.dropout(attn_output))
        stance_output = self.fc2(attn_output1)
        print("stance_output:",stance_output)

        return stance_output


# Define stance label mapping for majority vote in case of no decision
label_mapping = {"2": 0, "1": 1, "0": 2}

# read data 
data=pd.read_csv("../data/stance_train_clean.csv", delimiter=";") 
print("train data :: ",len(data))


train_ids=[]
train_texts=[]
train_comet_intents=[]
train_llm_intents=[]
train_stances=[]
train_llm1_stance=[]
train_llm2_stance=[]
train_llm1_rat=[]
train_llm2_rat=[]
train_targets=[]
train_aux_features=[]

for i in range(len(data)):
    id_=str(data.tweetid.values[i])
    target=(str(data.target.values[i]))
    train_ids.append(id_)
    train_texts.append(str(data.text.values[i]))
    train_targets.append(str(data.target.values[i]))
    l=ast.literal_eval(data.aux_features.values[i])
    train_aux_features.append(torch.tensor([l]))
    train_comet_intents.append(str(data.comet_intent.values[i]))
    train_llm_intents.append(str(data.llm_intent.values[i]))
    train_llm1_rat.append(str(data.gpt_rational.values[i]))
    train_llm2_rat.append(str(data.mistral_rational.values[i]))
    train_stances.append(str(data.true_stance.values[i]))
    train_llm1_stance.append(str(data.gpt_stance.values[i]))
    train_llm2_stance.append(str(data.mistral_stance.values[i]))
        
train_stances = [label_mapping[label] for label in train_stances]
train_llm1_stance = [label_mapping[label] for label in train_llm1_stance]
train_llm2_stance = [label_mapping[label] for label in train_llm2_stance]


test_ids=[]
test_texts=[]
test_stances=[]
test_targets=[]
test_comet_intents=[]
test_llm_intents=[]
test_llm1_stance=[]
test_llm2_stance=[]
test_llm1_rat=[]
test_llm2_rat=[]
test_aux_features=[]

data=pd.read_csv("../data/stance_test_clean.csv", delimiter=";") 
print("test data :: ",len(data))

for i in range(len(data)):
    id_=str(data.tweetid.values[i])
    target=(str(data.target.values[i]))
    seen=str(data.seen.values[i])
    if seen=="1":
        test_ids.append(id_)
        test_texts.append(str(data.text.values[i]))
        test_targets.append(str(data.target.values[i]))
        l=ast.literal_eval(data.aux_features.values[i])
        test_aux_features.append(torch.tensor([l]))
        test_comet_intents.append(str(data.comet_intent.values[i]))
        test_llm_intents.append(str(data.llm_intent.values[i]))
        test_llm1_rat.append(str(data.gpt_rational.values[i]))
        test_llm2_rat.append(str(data.mistral_rational.values[i]))
        test_stances.append(str(data.true_stance.values[i]))
        test_llm1_stance.append(str(data.gpt_stance.values[i]))
        test_llm2_stance.append(str(data.mistral_stance.values[i]))

test_stances = [label_mapping[label] for label in test_stances]
test_llm1_stance = [label_mapping[label] for label in test_llm1_stance]
test_llm2_stance = [label_mapping[label] for label in test_llm2_stance]
print("test data :: ",len(test_stances))

# Divide into train and validation tests
train_ids, val_ids, train_texts, val_texts, train_llm1_rat, val_llm1_rat, train_llm2_rat, val_llm2_rat, train_aux_features, val_aux_features,train_targets, val_targets, train_comet_intents, val_comet_intents,train_llm_intents,val_llm_intents, train_stances, val_stances, train_llm1_stance, val_llm1_stance, train_llm2_stance, val_llm2_stance = train_test_split(train_ids,train_texts, train_llm1_rat,train_llm2_rat, train_aux_features, train_targets,train_comet_intents,train_llm_intents,train_stances,train_llm1_stance,train_llm2_stance, test_size=0.15, random_state=42)

# BERT tokenizer
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

# Define datasets and data loaders
train_dataset = CreateDataset(train_ids, train_texts, train_llm1_rat, train_llm2_rat, train_aux_features, train_targets, train_comet_intents,train_llm_intents, train_stances, train_llm1_stance, train_llm2_stance, tokenizer, max_length=256)
val_dataset = CreateDataset(val_ids, val_texts, val_llm1_rat, val_llm2_rat, val_aux_features, val_targets, val_comet_intents,val_llm_intents, val_stances, val_llm1_stance, val_llm2_stance,tokenizer, max_length=256)
test_dataset = CreateDataset(test_ids, test_texts, test_llm1_rat, test_llm2_rat, test_aux_features, test_targets, test_comet_intents,test_llm_intents, test_stances, test_llm1_stance, test_llm2_stance,tokenizer, max_length=256)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)


# Model parameters
hidden_dim_rational = 768
rank = 256  # Rank for factorized bilinear pooling
num_epochs = 10
learning_rate = 0.01
num_heads=8
hidden_size=256
output_size=256
# Load BERT model
bert_model = AutoModel.from_pretrained("bert-base-uncased")
num_classes=3
dropout=0.5
num_intermediate_outputs=3
lambda_reg=0.1

# Load the saved Stage 1 Rational model
loaded_rational = RationalModel(bert_model,hidden_dim_rational, rank,num_heads,hidden_size,dropout,output_size)
loaded_rational.load_state_dict(torch.load("models/RationalModel.pth"))
loaded_rational.eval()    

# Load the saved Stage 1 Stance model
loaded_stance_stage1 = StanceModel_Stage1(hidden_size,num_classes)
loaded_stance_stage1.load_state_dict(torch.load("models/StanceModel_Stage1.pth"))
loaded_stance_stage1.eval()    
 
# Initialize Stage 2 Stance model
stance_model = StanceModel_Stage2(bert_model,rank,num_heads,hidden_size,num_intermediate_outputs,dropout,num_classes)

stance_criterion = nn.CrossEntropyLoss()
# Additional loss function for guiding the rational
gating_criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(stance_model.parameters(), lr=learning_rate)

# Training loop
for epoch in range(num_epochs):
    stance_model.train()
    total_loss = 0.0
    for ids,text_ids,text_mask,comet_intent_ids,comet_intent_mask,llm_intent_ids,llm_intent_mask,llm1_rational_binary,llm2_rational_binary,aux_feature,true_stance,llm1_stance,llm2_stance in train_loader:
               
        optimizer.zero_grad()
        loaded_rational_logits = loaded_rational(text_ids, text_mask, comet_intent_ids, comet_intent_mask, llm_intent_ids, llm_intent_mask)

        loaded_stance_logits=loaded_stance_stage1(loaded_rational_logits)
        loaded_rational_preds = (torch.sigmoid(loaded_rational_logits) > 0.5).int()

        # Forward pass for Stance task
        stance_logits = stance_model(text_ids,text_mask,aux_feature,loaded_rational_preds,llm1_rational_binary,llm2_rational_binary)
        
        # Calculate loss for Stance task
        stance_loss = stance_criterion(stance_logits, true_stance)
        
        # Compute additional loss associated with gating mechanism
        # L2 regularization on gating weights
        gating_loss_reg = torch.sum(stance_model.gated_intermediate.gating_weights** 2) 

        gating_weights=stance_model.gated_intermediate.gating_weights.unsqueeze(0)
        gating_weights=gating_weights.tolist()
        w1=gating_weights[0][0]
        w2=gating_weights[0][1]
        w3=gating_weights[0][2]
        gating_loss1=gating_criterion(stance_logits,loaded_stance_logits)
        gating_loss2=gating_criterion(stance_logits,llm1_stance)
        gating_loss3=gating_criterion(stance_logits,llm2_stance)
        gating_losses_weighted= w1*gating_loss1 + w2*gating_loss2 + w3*gating_loss3
        # print("gating_losses_weighted:::::",gating_losses_weighted,w1,w2,w3)

        # Total stance losses
        loss = stance_loss + lambda_reg*gating_loss_reg + gating_losses_weighted 
        
        # Backward pass
        loss.backward()
        optimizer.step()
        total_loss += loss.item()


    # Validation loop
    stance_model.eval()
    val_losses = []
    val_predictions = []
    val_targets = []
    with torch.no_grad():
        for ids,text_ids,text_mask,comet_intent_ids,comet_intent_mask,llm_intent_ids,llm_intent_mask,llm1_rational_binary,llm2_rational_binary,aux_feature,true_stance,llm1_stance,llm2_stance in val_loader:
            
            loaded_rational_logits = loaded_rational(text_ids, text_mask, comet_intent_ids, comet_intent_mask, llm_intent_ids, llm_intent_mask)
            loaded_stance_logits=loaded_stance_stage1(loaded_rational_logits)
            loaded_rational_preds = (torch.sigmoid(loaded_rational_logits) > 0.5).int()

            # Forward pass for Stance task
            stance_logits = stance_model(text_ids,text_mask,aux_feature,loaded_rational_preds,llm1_rational_binary,llm2_rational_binary)
            
            # Calculate loss for Stance task
            stance_loss = stance_criterion(stance_logits, true_stance)

            # Compute additional loss associated with gating mechanism
            # L2 regularization on gating weights
            gating_loss_reg = torch.sum(stance_model.gated_intermediate.gating_weights** 2)

            gating_weights=stance_model.gated_intermediate.gating_weights.unsqueeze(0)
            gating_weights=gating_weights.tolist()
            w1=gating_weights[0][0]
            w2=gating_weights[0][1]
            w3=gating_weights[0][2]
            # Gating loss as per predicted stance using rationals per gating weight
            gating_loss1=gating_criterion(stance_logits,loaded_stance_logits)
            gating_loss2=gating_criterion(stance_logits,llm1_stance)
            gating_loss3=gating_criterion(stance_logits,llm2_stance)
            gating_losses_weighted= w1*gating_loss1 + w2*gating_loss2 + w3*gating_loss3
            
            # Only stance losses
            val_loss = stance_loss + lambda_reg*gating_loss_reg + gating_losses_weighted
            val_losses.append(val_loss.item())
            
            # Calculate accuracy
            pred=torch.argmax(stance_logits, dim=1).cpu().numpy()

            val_predictions.extend(torch.argmax(stance_logits, dim=1).cpu().numpy())
            val_targets.extend(true_stance.cpu().numpy())

    val_accuracy = accuracy_score(val_targets, val_predictions)
    avg_val_loss = sum(val_losses) / len(val_losses)
    print(f"Epoch {epoch+1}/{num_epochs}, Validation Loss: {avg_val_loss}, Stance Validation Accuracy: {val_accuracy}")

# Save the trained model
torch.save(stance_model.state_dict(), "models/Stage2_Stance.pth")

loaded_stance_stage2 = StanceModel_Stage2(bert_model,rank,num_heads,hidden_size,num_intermediate_outputs,dropout,num_classes)
loaded_stance_stage2.load_state_dict(torch.load("models/Stage2_Stance.pth"))

############ evaluate stance model
loaded_stance_stage2.eval()
test_predictions = []
test_targets = []
test_predictions = []
test_targets = []
test_probabilities = []
test_texts_list = []
test_ids_list=[]

with torch.no_grad():
    for ids,text_ids,text_mask,comet_intent_ids,comet_intent_mask,llm_intent_ids,llm_intent_mask,llm1_rational_binary,llm2_rational_binary,aux_feature,true_stance,llm1_stance,llm2_stance in test_loader:
        
        loaded_rational_logits = loaded_rational(text_ids, text_mask, comet_intent_ids, comet_intent_mask, llm_intent_ids, llm_intent_mask)

        loaded_stance_logits=loaded_stance_stage1(loaded_rational_logits)
        loaded_rational_preds = (torch.sigmoid(loaded_rational_logits) > 0.5).int()

        stance_logits = loaded_stance_stage2(text_ids,text_mask,aux_feature,loaded_rational_preds,llm1_rational_binary,llm2_rational_binary)

        probabilities = torch.softmax(stance_logits, dim=-1).cpu().numpy()
        predictions = torch.argmax(stance_logits, dim=1).cpu().numpy()
        test_predictions.extend(predictions)
        test_targets.extend(true_stance)
        test_probabilities.extend(probabilities)

print("test_targets:",test_targets)
print("test_predictions",test_predictions)
test_accuracy = accuracy_score(test_targets, test_predictions)
print(f"Test Accuracy: {test_accuracy}")

# Save predictions, probabilities, and test texts into a CSV file
with open("predictions/Stance.csv", "w", newline="", encoding="utf-8") as csvfile:
    fieldnames = ["True", "Prediction", "Probability_None", "Probability_Favor", "Probability_Against"]
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    writer.writeheader()

    for text, prediction, probabilities in zip(test_targets, test_predictions, test_probabilities):
        writer.writerow({
            "True": text,
            "Prediction": prediction,
            "Probability_None": probabilities[0],
            "Probability_Favor": probabilities[1],
            "Probability_Against": probabilities[2]
        })

target_names =['2',"1","0"]
class_rep=classification_report(test_targets, test_predictions, target_names=target_names,digits=4)
print("class_rep",class_rep)
