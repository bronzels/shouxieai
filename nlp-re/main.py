import os
import time
import torch
from tqdm import tqdm
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from sklearn import metrics

from utils import SentenceREDataset, load_checkpoint, save_checkpoint
from model import SentenceRE
from config import hparams
from split_data import split_data

def train(hparams):
    train_file = hparams.train_file
    dev_file = hparams.dev_file
    test_file = hparams.test_file
    if not os.path.exists(train_file):
        split_data(hparams.all_data)

    device = hparams.device
    bert_path = hparams.bert_path

    target_file = hparams.target_file
    log_dir = hparams.log_dir
    if not os.path.exists(log_dir):
        os.mkdir(log_dir)
        os.mkdir(os.path.split(hparams.model_best_bin)[0][2:])

    model_best_bin = hparams.model_best_bin
    model_best_checkpoint_bin = hparams.model_best_checkpoint_bin
    model_last_bin = hparams.model_last_bin
    model_last_checkpoint_bin = hparams.model_last_checkpoint_bin

    batch_size = hparams.batch_size
    epochs = hparams.epochs

    learning_rate = hparams.learning_rate
    weight_decay = hparams.weight_decay

    train_dataset = SentenceREDataset(train_file, target_path=target_file, pretrained_model_path=bert_path)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=train_dataset.collate_fn)

    dev_dataset = SentenceREDataset(dev_file, target_path=target_file, pretrained_model_path=bert_path)
    dev_loader = DataLoader(dev_dataset, batch_size=batch_size, shuffle=False, collate_fn=dev_dataset.collate_fn)

    test_dataset = SentenceREDataset(test_file, target_path=target_file, pretrained_model_path=bert_path)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, collate_fn=test_dataset.collate_fn)

    model = SentenceRE(hparams).to(device)

    if os.path.exists(model_best_checkpoint_bin):
        checkpoint_dict = load_checkpoint(model_best_checkpoint_bin)
        best_f1 = checkpoint_dict['best_f1']
        epoch_offset = checkpoint_dict['best_epoch'] + 1
        model.load_state_dict(torch.load(model_best_bin))
    else:
        checkpoint_dict = {}
        best_f1 = 0.0
        epoch_offset = 0

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    criterion = torch.nn.CrossEntropyLoss().to(device)
    writer = SummaryWriter(os.path.join(log_dir, time.strftime("%Y-%m-%d_%H-%M-%S", time.localtime())))

    for epoch in range(epoch_offset, epochs):
        model.train()
        msg = ""
        train_loss, train_loss_agv = 0, 0
        for i_batch, sample_batched in enumerate(tqdm(train_loader, desc='Training')):
            token_ids = sample_batched['input_ids'].to(device)
            token_type_ids = sample_batched['token_type_ids'].to(device)
            attention_mask = sample_batched['attention_mask'].to(device)
            e1_mask = sample_batched['e1_masks'].to(device)
            e2_mask = sample_batched['e2_masks'].to(device)
            tag_ids = sample_batched['labels'].to(device)

            model.zero_grad()
            preds = model(token_ids, token_type_ids, attention_mask, e1_mask, e2_mask)
            loss = criterion(preds, tag_ids)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            train_loss_agv += loss.item()
            if i_batch % 10 == 9:
                writer.add_scalar('Training/training loss', train_loss / 10, epoch * len(train_loader) + i_batch)
                train_loss = 0.0

        msg = f"[{epoch}/{epochs}] train_loss: {train_loss_agv / len(train_loader):.4f}"

        if epoch % 2 == 0:
            model.eval()
            tags_true, tags_pred = [], []
            save_best = False
            with torch.no_grad():
                for val_i_batch, val_sample_batched in enumerate(tqdm(dev_loader, desc='valid')):
                    token_ids = val_sample_batched['input_ids'].to(device)
                    token_type_ids = val_sample_batched['token_type_ids'].to(device)
                    attention_mask = val_sample_batched['attention_mask'].to(device)
                    e1_mask = val_sample_batched['e1_masks'].to(device)
                    e2_mask = val_sample_batched['e2_masks'].to(device)
                    tag_ids = val_sample_batched['labels']

                    preds = model(token_ids, token_type_ids, attention_mask, e1_mask, e2_mask)
                    preds_tag_ids = preds.argmax(1)
                    tags_true.extend(tag_ids.tolist())
                    tags_pred.extend(preds_tag_ids.tolist())

            f1 = metrics.f1_score(tags_true, tags_pred, average='weighted')
            precision = metrics.precision_score(tags_true, tags_pred, average='weighted')
            recall = metrics.recall_score(tags_true, tags_pred, average='weighted')
            accuracy = metrics.accuracy_score(tags_true, tags_pred)

            writer.add_scalar('valid/f1', f1, epoch)
            writer.add_scalar('valid/precision', precision, epoch)
            writer.add_scalar('valid/recall', recall, epoch)
            writer.add_scalar('valid/accuracy', accuracy, epoch)

            if checkpoint_dict.get('epoch_f1'):
                checkpoint_dict['epoch_f1'][epoch] = f1
            else:
                checkpoint_dict['epoch_f1'] = {epoch: f1}
            if f1 > best_f1:
                best_f1 = f1
                checkpoint_dict['best_f1'] = best_f1
                checkpoint_dict['best_epoch'] = epoch
                torch.save(model.state_dict(), model_best_bin)
                save_best = True
            save_checkpoint(checkpoint_dict, model_best_checkpoint_bin)

            msg += f" dev_acc: {accuracy:.4f} dev_f1: {f1:.4f} {'*' if save_best else ''}"

        torch.save(model.state_dict(), model_last_bin)
        save_checkpoint(checkpoint_dict, model_last_checkpoint_bin)

        print(msg)

    model.eval()
    tags_true, tags_pred = [], []
    with torch.no_grad():
        for sample_batched in tqdm(test_loader, desc='test'):
            token_ids = sample_batched['input_ids'].to(device)
            token_type_ids = sample_batched['token_type_ids'].to(device)
            attention_mask = sample_batched['attention_mask'].to(device)
            e1_mask = sample_batched['e1_masks'].to(device)
            e2_mask = sample_batched['e2_masks'].to(device)
            tag_ids = sample_batched['labels']

            preds = model(token_ids, token_type_ids, attention_mask, e1_mask, e2_mask)
            preds_tag_ids = preds.argmax(1)
            tags_true.extend(tag_ids.tolist())
            tags_pred.extend(preds_tag_ids.tolist())

    accuracy = metrics.accuracy_score(tags_true, tags_pred)
    print(f"last epoch: test acc: {accuracy: .4f}")
    writer.add_scalar('test acc', accuracy)
    writer.close()

if __name__ == "__main__":
    train(hparams)

"""
epoch=2
        return re_tokens[1:-1], pos1, pos2
处理数据 train: 100%|██████████| 120000/120000 [00:33<00:00, 3621.11it/s]
处理数据 val: 100%|██████████| 40000/40000 [00:11<00:00, 3615.04it/s]
处理数据 test: 100%|██████████| 40000/40000 [00:11<00:00, 3621.77it/s]
Some weights of the model checkpoint at ./bert-base-chinese were not used when initializing BertModel: ['cls.predictions.transform.dense.weight', 'cls.seq_relationship.bias', 'cls.predictions.decoder.weight', 'cls.seq_relationship.weight', 'cls.predictions.transform.LayerNorm.weight', 'cls.predictions.transform.LayerNorm.bias', 'cls.predictions.transform.dense.bias', 'cls.predictions.bias']
- This IS expected if you are initializing BertModel from the checkpoint of a model trained on another task or with another architecture (e.g. initializing a BertForSequenceClassification model from a BertForPreTraining model).
- This IS NOT expected if you are initializing BertModel from the checkpoint of a model that you expect to be exactly identical (initializing a BertForSequenceClassification model from a BertForSequenceClassification model).
Training: 100%|██████████| 1875/1875 [15:16<00:00,  2.05it/s]
valid: 100%|██████████| 625/625 [01:03<00:00,  9.77it/s]
Training:   0%|          | 0/1875 [00:00<?, ?it/s][0/2] train_loss: 0.8859 dev_acc: 0.8025 dev_f1: 0.7984 *
Training: 100%|██████████| 1875/1875 [15:20<00:00,  2.04it/s]
[1/2] train_loss: 0.4653
test: 100%|██████████| 625/625 [01:03<00:00,  9.79it/s]
last epoch: test acc:  0.8799
        return re_tokens, pos1, pos2
处理数据 train: 100%|██████████| 120000/120000 [00:33<00:00, 3599.48it/s]
处理数据 val: 100%|██████████| 40000/40000 [00:11<00:00, 3591.06it/s]
处理数据 test: 100%|██████████| 40000/40000 [00:11<00:00, 3608.03it/s]
Some weights of the model checkpoint at ./bert-base-chinese were not used when initializing BertModel: ['cls.predictions.transform.dense.bias', 'cls.seq_relationship.weight', 'cls.predictions.decoder.weight', 'cls.predictions.bias', 'cls.predictions.transform.LayerNorm.weight', 'cls.predictions.transform.LayerNorm.bias', 'cls.predictions.transform.dense.weight', 'cls.seq_relationship.bias']
- This IS expected if you are initializing BertModel from the checkpoint of a model trained on another task or with another architecture (e.g. initializing a BertForSequenceClassification model from a BertForPreTraining model).
- This IS NOT expected if you are initializing BertModel from the checkpoint of a model that you expect to be exactly identical (initializing a BertForSequenceClassification model from a BertForSequenceClassification model).
Training: 100%|██████████| 1875/1875 [15:41<00:00,  1.99it/s]
[1/2] train_loss: 0.4722
test: 100%|██████████| 625/625 [01:05<00:00,  9.48it/s]
last epoch: test acc:  0.8814


"""