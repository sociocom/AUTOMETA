
import pandas as pd
import numpy as np
from tqdm import trange

import torch, random
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from transformers import BertTokenizer, BertConfig
# from transformers import BertForTokenClassification,
from transformers import AdamW
from transformers import get_linear_schedule_with_warmup
from seqeval.metrics import f1_score, accuracy_score

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report

from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split

from transformers import LongformerTokenizer, LongformerForTokenClassification
from transformers import get_linear_schedule_with_warmup
from seqeval.metrics import f1_score, accuracy_score

import csv
import itertools

seed_val=50
random.seed(seed_val)

import os

import warnings
warnings.filterwarnings("ignore", category=UserWarning)


torch.cuda.empty_cache()




class SentenceGetter(object):

    def __init__(self, data):
        self.n_sent = 1
        self.data = data
        self.empty = False
        agg_func = lambda s: [(w, t, p) for w, t, p in zip(s["Word"].values.tolist(),s["Tag"].values.tolist(),s["pmid"].values.tolist())]
        self.grouped = self.data.groupby("pmid").apply(agg_func)
        self.sentences = [s for s in self.grouped]

    def get_next(self):
        try:
            s = self.grouped["Sentence: {}".format(self.n_sent)]
            self.n_sent += 1
            return s
        except:
            return None


def create_train_data(train_file,MAX_LEN,bs):
    data = pd.read_csv(train_file, encoding="utf-8-sig", header=None,names=['pmid','Word','Tag'])
    data['Word'] = data['Word'].astype(str)
    data['pmid'] = data['pmid'].astype(int)
    data['Tag'] = data['Tag'].astype(str)


    getter = SentenceGetter(data)
    sentences = [[word[0] for word in sentence] for sentence in getter.sentences]

    labels = [[s[1] for s in sentence] for sentence in getter.sentences]
    pmids=[[s[2] for s in sentence] for sentence in getter.sentences]


    tag_values = ['O', 'PAD', 'total-participants', 'control-participants', 'intervention-participants', 'age',
                  'ethinicity',
                  'location', 'eligibility', 'condition', 'intervention', 'control', 'outcome-measure', 'outcome',
                  'iv-bin-abs', 'iv-bin-percent', 'iv-cont-mean', 'iv-cont-sd', 'iv-cont-median', 'iv-cont-q1',
                  'iv-cont-q3',
                  'cv-bin-abs', 'cv-bin-percent', 'cv-cont-mean', 'cv-cont-sd', 'cv-cont-median', 'cv-cont-q1',
                  'cv-cont-q3',
                  'control-value-continous']


    tag2idx = {t: i for i, t in enumerate(tag_values)}


    tokenized_texts_and_labels = [tokenize_and_preserve_labels(sent, labs, pids)for sent, labs, pids in zip(sentences, labels, pmids)]

    tokenized_texts = [token_label_pair[0] for token_label_pair in tokenized_texts_and_labels]

    labels = [token_label_pair[1] for token_label_pair in tokenized_texts_and_labels]
    pmids = [token_label_pair[2] for token_label_pair in tokenized_texts_and_labels]

    input_ids = pad_sequences([tokenizer.convert_tokens_to_ids(txt) for txt in tokenized_texts],maxlen=MAX_LEN, dtype="long",
                              value=0.0,truncating="post", padding="post")

    tags = pad_sequences([[tag2idx.get(l) for l in lab] for lab in labels],maxlen=MAX_LEN, dtype="long",
                         value=tag2idx['PAD'], truncating="post", padding="post")

    pmids_tag= pad_sequences([id for id in pmids],maxlen=MAX_LEN, value=5, padding="post",dtype="long", truncating="post")

    attention_masks = [[float(i != 0.0) for i in ii] for ii in input_ids]


    tr_inputs=input_ids
    tr_tags=tags
    tr_pmid=pmids_tag
    tr_masks=attention_masks


    tr_inputs = torch.tensor(tr_inputs)
    tr_tags = torch.tensor(tr_tags)
    tr_pmid=torch.tensor(tr_pmid)

    tr_masks = torch.tensor(tr_masks)


    train_data = TensorDataset(tr_inputs, tr_masks, tr_tags, tr_pmid)
    train_dataloader = DataLoader(train_data, sampler=None, batch_size=bs, shuffle=False)

    return tag2idx, tag_values, train_dataloader, tokenized_texts, pmids




def tokenize_and_preserve_labels(sentence, text_labels, pmid):
    tokenized_sentence = []
    labels = []
    pmids= []

    i=0
    for word, label, pid in zip(sentence, text_labels, pmid):
        i+=1
        if i == 1:
            tokenized_word = tokenizer.tokenize(word)
            n_subwords = len(tokenized_word)
            tokenized_sentence.extend(tokenized_word)
            labels.extend([label] * n_subwords)
            pmids.extend([pid] * n_subwords)

        else:
            tokenized_word = tokenizer.tokenize(word)
            tokenized_word[0] = 'Ġ' + tokenized_word[0]
            n_subwords = len(tokenized_word)
            tokenized_sentence.extend(tokenized_word)
            labels.extend([label] * n_subwords)
            pmids.extend([pid] * n_subwords)
            
    return tokenized_sentence, labels, pmids





def model_training (epochs, learning_rate, batch_size,tr_file,te_file,MAX_LEN,mpath,saveFolder):
    hyperparameterFile=open(os.path.join(saveFolder,'longFormer_paramsCheck.csv'),'a')
    writer=csv.writer(hyperparameterFile,delimiter=',')
    writer.writerow(['epochs','batch','learning_rate','accuracy','F1'])
    
    _, _, train_dataloader, tr_tokenized_text,_= create_train_data(tr_file, bs=batch_size,MAX_LEN=MAX_LEN)
    tag2idx, tag_values,valid_dataloader, te_tokenized_text, te_pid =  create_train_data(te_file,bs=batch_size, MAX_LEN=MAX_LEN)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # try:
    #     print('starting with existing model!')
    #     model=torch.load(os.path.join(saveFolder,'bestmodel.pt'))
    # except:
    #     print('did not load existing model!')
    model = LongformerForTokenClassification.from_pretrained(mpath,num_labels=len(tag2idx) ,output_attentions = False,output_hidden_states = False)
    model.cuda()
    
    
    
#     FULL_FINETUNING = True
#     if FULL_FINETUNING:
    param_optimizer = list(model.named_parameters())
    no_decay = ['bias', 'gamma', 'beta']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
         'weight_decay_rate': 0.01},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)],
         'weight_decay_rate': 0.0}]
#     else:
#     param_optimizer = list(model.classifier.named_parameters())
#     optimizer_grouped_parameters = [{"params": [p for n, p in param_optimizer]}]

    optimizer = AdamW(optimizer_grouped_parameters,lr=learning_rate, eps=1e-8)

    ################################
    # params = model.parameters()
    # optimizer = torch.optim.Adam(params, lr=learning_rate)
    #################################
    
    max_grad_norm = 1.0
    total_steps = len(train_dataloader) * epochs
    scheduler = get_linear_schedule_with_warmup( optimizer,num_warmup_steps=0,num_training_steps=total_steps)


    loss_values, validation_loss_values = [], []
    accuracy_vals, F1_vals = [], []
    
    best_acc=0
    best_F=0
    
    classR_directory=os.path.join(saveFolder,'classification_reports')
    try: 
        os.mkdir(classR_directory)
    except:
        pass
                 
    teForecast_directory=os.path.join(saveFolder,'test_forecasts')
    try: 
        os.mkdir(teForecast_directory)
    except:
        pass
    
    
    for ep in range(epochs):
        print('\n***********  training epoch ',ep, '****************************')
        model.train()
        total_loss = 0
        
        for step, batch in enumerate(train_dataloader):
            batch = tuple(t.to(device) for t in batch)
            b_input_ids, b_input_mask, b_labels, b_pids = batch
            model.zero_grad()
            outputs = model(b_input_ids, token_type_ids=None,attention_mask=b_input_mask, labels=b_labels)

            loss = outputs[0]
            loss.backward()
            total_loss += loss.item()
            torch.nn.utils.clip_grad_norm_(parameters=model.parameters(), max_norm=max_grad_norm)
            optimizer.step()
            scheduler.step()
        
        avg_train_loss = total_loss / len(train_dataloader)
        print("Average train loss: {}".format(avg_train_loss))
        loss_values.append(avg_train_loss)


        # ========================================
        #               Validation
        # ========================================

        model.eval()
        
        eval_loss, eval_accuracy = 0, 0
        nb_eval_steps, nb_eval_examples = 0, 0
        predictions , true_labels = [], []
        
        pmidss=[]
        for batch in valid_dataloader:
            batch = tuple(t.to(device) for t in batch)
            b_input_ids, b_input_mask, b_labels, b_pids = batch
            with torch.no_grad():
                outputs = model(b_input_ids, token_type_ids=None,attention_mask=b_input_mask, labels=b_labels)

            logits = outputs[1].detach().cpu().numpy()
            label_ids = b_labels.to('cpu').numpy()

            eval_loss += outputs[0].mean().item()
            predictions.extend([list(p) for p in np.argmax(logits, axis=2)])
            true_labels.extend(label_ids)
            
            label_pmid=b_pids.to('cpu').numpy()
            pmidss.extend(label_pmid)

        eval_loss = eval_loss / len(valid_dataloader)
        validation_loss_values.append(eval_loss)
        print("Validation loss: {}".format(eval_loss))
        
        pred_tags = [tag_values[p_i] for p, l in zip(predictions, true_labels)
                                     for p_i, l_i in zip(p, l) if tag_values[l_i] != "PAD"]
        valid_tags = [tag_values[l_i] for l in true_labels
                                      for l_i in l if tag_values[l_i] != "PAD"]
        
        
        
        
        print("Validation Accuracy: {}".format(accuracy_score(pred_tags, valid_tags)))
        print("Validation F1-Score: {}".format(f1_score([pred_tags], [valid_tags])))

        #saving classification report
        report=classification_report(valid_tags, pred_tags,output_dict = True)
        # print(classification_report(valid_tags, pred_tags))

        ###saving best model
        accuracy_vals.append(accuracy_score(pred_tags, valid_tags)), F1_vals.append(f1_score([pred_tags], [valid_tags]))

        val_F1=f1_score([pred_tags], [valid_tags])
        
        writer.writerow([ep,batch_size,learning_rate,accuracy_score(pred_tags, valid_tags),val_F1])
        hyperparameterFile.flush()
        
        if val_F1>=best_F:
            best_F=val_F1
            torch.save(model,os.path.join(saveFolder,'bestmodel.pt'))
            report_df = pd.DataFrame(report).T
            report_df.to_csv(os.path.join(classR_directory,'longformer_best_classification_report.csv'))
            

            for i in range(len(te_tokenized_text)):
                p, t, id= predictions[i], true_labels[i],pmidss[i]
                preds=[tag_values[p_i] for p_i, l_i in zip(p, t) if tag_values[l_i] != "PAD"]
                valids=[tag_values[l_i] for l_i in t if tag_values[l_i] != "PAD"]
                idss = [pi for pi in id if pi != 5]

                df_pid = te_pid[i]
                df_tokens = te_tokenized_text[i]
                df_true_labels = valids
                df_pred_labes = preds

                Npid, Ntoken, NtrueL, NpredL = [], [], [], []
                for i,(pid, tk, tl, pl) in enumerate(zip(df_pid, df_tokens, df_true_labels, df_pred_labes)):
                    tk = str(tk)
                    if i == 0:
                        tk = 'Ġ' + tk

                    if not tk.startswith('Ġ'):
                        Ntoken[-1] = Ntoken[-1] + tk
                    else:
                        tk = tk.replace('Ġ', '')
                        Ntoken.append(tk)
                        Npid.append(pid)
                        NtrueL.append(tl)
                        NpredL.append(pl)

                   
                newDf = pd.DataFrame()
                newDf['pmid'] = np.array(Npid)
                newDf['token'] = np.array(Ntoken)
                newDf['trueL'] = np.array(NtrueL)
                newDf['predL'] = np.array(NpredL)
                
                
                newDf.to_csv(os.path.join(teForecast_directory,str(df_pid[0])+'_forecast.csv'),index=False)
        print()

    hyperparameterFile.close()
    print('round finished----------')
  

tr_file='../train_file.csv'
te_file='../test_file.csv'


longFmodelpath='allenai/longformer-large-4096'
longFmodelname=longFmodelpath.split('/')[-1]

tokenizer = LongformerTokenizer.from_pretrained(longFmodelpath)

longFModeldirectory='../longformer_model_results'
longFModeldirectory=os.path.join(longFModeldirectory, str(seed_val))
try:
    os.mkdir(longFModeldirectory)
except:
    pass

longFModeldirectory=os.path.join(longFModeldirectory, longFmodelname)
try:
    os.mkdir(longFModeldirectory)
except:
    pass



epochs=[10]
batch=[2]
lr= [0.00007]

params=[epochs, batch, lr]
params_combine=list(itertools.product(*params))


params_Set=[{'epochs':i, 'batch':j,'lr':k} for (i,j,k) in params_combine]

for param_set in params_Set:
    print('--------param_set-----------', param_set)
    try:
        model_training(epochs = param_set['epochs'], learning_rate = param_set['lr'],batch_size = param_set['batch'],MAX_LEN = 1500, tr_file=tr_file,te_file=te_file, saveFolder=longFModeldirectory, mpath=longFmodelpath)
    except Exception as e:
        print('****************skipped part: ', param_set),print(e)
        continue






