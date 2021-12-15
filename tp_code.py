import torch
import transformers

# Managing arrays
import numpy as np

# load the TensorBoard notebook extension
# %load_ext tensorboard

if torch.cuda.is_available():
  print("GPU is available.")
  device = torch.cuda.current_device()
else:
  print("Will work on CPU.")
  
  
## DATA

from sklearn.datasets import fetch_20newsgroups

categories = [
 'comp.windows.x',
 'sci.med',
 'soc.religion.christian',
 'talk.politics.guns',
]

# download data if not already present in data_home
trainset = fetch_20newsgroups(subset='train', categories=categories, shuffle=True, random_state=42, data_home='./scikit_learn_data')
testset = fetch_20newsgroups(subset='test', categories=categories, shuffle=True, random_state=42, data_home='./scikit_learn_data')

# define input data and labels for training and testing
x_train = trainset.data
y_train = trainset.target
x_test = testset.data
y_test = testset.target

# # SOLUTION (yes, we are cool)
print('taille du jeu de données: \n \t {} posts de forum en total'.format(len(x_train) + len(x_test)))
print('\t {} posts pour le training'.format(len(x_train)))
for i in range(len(categories)):
  num = sum(y_train == i)
  print("\t\t {} {}".format(num, categories[i]))
  print('\t {} posts pour le test'.format(len(x_test)))
for i in range(len(categories)):
  num = sum(y_test == i)
  print("\t\t {} {}".format(num, categories[i]))

print('\n')
print('EXEMPLE: \n')
print(x_train[0])


def clean_post(post: str, remove_start: tuple):
    clean_lines = []
    for line in post.splitlines():
            if not line.startswith(remove_start):
                clean_lines.append(line)
    return '\n'.join(clean_lines)
    

# SOLUTION (yes, again, we are cool)
remove_start = (
  'From:',
  'Subject:',
  'Reply-To:',
  'In-Reply-To:',
  'Nntp-Posting-Host:',
  'Organization:',
  'X-Mailer:',
  'In article <',
  'Lines:',
  'NNTP-Posting-Host:',
  'Summary:',
  'Article-I.D.:'
)
x_train = [clean_post(p, remove_start) for p in x_train]
x_test = [clean_post(p, remove_start) for p in x_test]



## TOKENISATION

from transformers import DistilBertTokenizer

MAX_LEN = 512

tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased', padding=True, truncation=True)

# let's check out how the tokenizer works
for n in range(3):
    # tokenize forum post
    tokenizer_out = tokenizer(x_train[n])
    # convert numerical tokens to alphabetical tokens
    encoded_tok = tokenizer.convert_ids_to_tokens(tokenizer_out.input_ids)
    # decode tokens back to string
    decoded = tokenizer.decode(tokenizer_out.input_ids)
    print(tokenizer_out)
    print(encoded_tok, '\n')
    print(decoded, '\n')
    print('---------------- \n')


from torch.utils.data import Dataset, DataLoader

MAX_LEN = 512

class PostsDataset(Dataset):
    def __init__(self, posts, labels, tokenizer, max_len):
        # variables that are set when the class is instantiated
        self.posts = posts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.posts)
  
    def __getitem__(self, item):
        # select the post and its category
        post = str(self.posts[item])
        label = self.labels[item]
        # tokenize the post
        tokenizer_out = self.tokenizer(
            post,
            add_special_tokens=True,
            max_length=self.max_len,
            return_token_type_ids=False,
            pad_to_max_length=True,
            return_attention_mask=True,
            return_tensors='pt',
            truncation=True
            )
        # return a dictionary with the output of the tokenizer and the label
        return  {
            'input_ids': tokenizer_out['input_ids'].flatten(),
            'attention_mask': tokenizer_out['attention_mask'].flatten(),
            'label': torch.tensor(label, dtype=torch.long)
        }


# instantiate two PostsDatasets
train_dataset = PostsDataset(x_train, y_train, tokenizer, MAX_LEN)
test_dataset = PostsDataset(x_test, y_test, tokenizer, MAX_LEN)


## MODEL

from transformers import DistilBertModel

PRE_TRAINED_MODEL_NAME = 'distilbert-base-uncased'

distilbert = DistilBertModel.from_pretrained(PRE_TRAINED_MODEL_NAME)
 
first_post = train_dataset[0]

hidden_state = distilbert(
    input_ids=first_post['input_ids'].unsqueeze(0), attention_mask=first_post['attention_mask'].unsqueeze(0)
    )

print(hidden_state[0].shape)

print(distilbert.config)


from transformers import DistilBertPreTrainedModel, DistilBertConfig


PRE_TRAINED_MODEL_NAME = 'distilbert-base-uncased'

class DistilBertForPostClassification(DistilBertPreTrainedModel):
    def __init__(self, config, num_labels, freeze_encoder=False):
        # instantiate the parent class DistilBertPreTrainedModel
        super().__init__(config)
        # instantiate num. of classes
        self.num_labels = num_labels
        # instantiate and load a pretrained DistilBERT model as encoder
        self.encoder = DistilBertModel.from_pretrained(PRE_TRAINED_MODEL_NAME)
        # freeze the encoder parameters if required (Q1)
        if freeze_encoder:
          for param in self.encoder.parameters():
              param.requires_grad = False
        # the classifier: a feed-forward layer attached to the encoder's head
        self.classifier = torch.nn.Linear(
            in_features=config.dim, out_features=self.num_labels, bias=True)
        # instantiate a dropout function for the classifier's input
        self.dropout = torch.nn.Dropout(p=0.1)


    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
        output_attentions=None,
        output_hidden_states=None,
    ):
        # encode a batch of sequences with DistilBERT
        encoder_output = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
        )
        # extract the hidden representations from the encoder output
        hidden_state = encoder_output[0]  # (bs, seq_len, dim)
        # only select the encoding corresponding to the first token
        # of each sequence in the batch (Q2)
        pooled_output = hidden_state[:, 0]  # (bs, dim)
        # apply dropout
        pooled_output = self.dropout(pooled_output)  # (bs, dim)
        # feed into the classifier
        logits = self.classifier(pooled_output)  # (bs, dim)

        outputs = (logits,) + encoder_output[1:]
        
        if labels is not None: # (Q3)
          # instantiate loss function
          # SOLUTION : loss_fct = torch.nn.CrossEntropyLoss()
          # calculate loss
          # SOLUTION : loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
          # aggregate outputs
          outputs = (loss,) + outputs

        return outputs  # (loss), logits, (hidden_states), (attentions)


# instantiate model
model = DistilBertForPostClassification(
    config=distilbert.config, num_labels=len(categories), freeze_encoder = True
    )

# print info about model's parameters
total_params = sum(p.numel() for p in model.parameters())
model_parameters = filter(lambda p: p.requires_grad, model.parameters())
trainable_params = sum([np.prod(p.size()) for p in model_parameters])
print('model total params: ', total_params)
print('model trainable params: ', trainable_params)
print('\n', model)


## TRAINING

from transformers import Trainer, TrainingArguments
from sklearn.metrics import precision_recall_fscore_support, accuracy_score

def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds)
    acc = accuracy_score(labels, preds)
    return {
        'accuracy': acc,
        'f1': f1,
        'precision': precision,
        'recall': recall
    }

training_args = TrainingArguments(
    output_dir='./results',          
    logging_dir='./logs',
    logging_first_step=True,
    logging_steps=50,
    num_train_epochs=4,              
    per_device_train_batch_size=8,  
    learning_rate=5e-5,
    weight_decay=0.01        
)

trainer = Trainer(
    model=model,                         
    args=training_args,                  
    train_dataset=train_dataset,         
    compute_metrics=compute_metrics
)

train_results = trainer.train()

test_results = trainer.predict(test_dataset=test_dataset)

print('Predictions: \n', test_results.predictions)
print('\nAccuracy: ', test_results.metrics['eval_accuracy'])
print('Precision: ', test_results.metrics['eval_precision'])
print('Recall: ', test_results.metrics['eval_recall'])
print('F1: ', test_results.metrics['eval_f1'])
print(categories)

MODEL_PATH = './my_model'
trainer.save_model(MODEL_PATH)


## PREDICTIONS

model = DistilBertForPostClassification.from_pretrained(
    './my_model', config=distilbert.config, num_labels=len(categories)).to(device)
for sentence in ['Lung cancer is a deadly disease.', 'God is love', 'How can you install Microsoft Office extensions?', 'Gun killings increase every year.']:
  encoding = tokenizer.encode_plus(sentence)
  encoding['input_ids'] = torch.tensor([encoding.input_ids]).to(device)
  encoding['attention_mask'] = torch.tensor(encoding.attention_mask).to(device)
  out = model(**encoding)
  categories_probability = torch.nn.functional.softmax(out[0], dim=1).flatten()
  print(sentence)
  print('\tProbabilities assigned by the model : ')
  for n,c in enumerate(categories):
    print('\t\t{} : {}'.format(c, categories_probability[n]))
  print('\n\t--> Prediction :', categories[categories_probability.argmax()])
  print('------------------------------------------------\n')
  
  
  
## IMPROVE THE MODEL

# # SOLUTION 1 (trivial): increase training epochs
# # SOLUTION 2: finetune encoder parameters too

# model_unfreezed = DistilBertForPostClassification(config, freeze_decoder = False)
# trainer_unfreezed = Trainer(
#     model=model_unfreezed,                         
#     args=training_args,                  
#     train_dataset=train_dataset,         
#     compute_metrics=compute_metrics
# )
# trainer_unfreezed.train()
# trainer_unfreezed.predict(test_dataset=test_dataset)

# # SOLUTION 3: let's see what students can do !
