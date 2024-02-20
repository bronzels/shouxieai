import utils
import config
import logging
import numpy as np
from data_process import Processor
from data_loader import NERDataset
from model import BertNER
from train import train, evaluate

from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from transformers.optimization import get_cosine_schedule_with_warmup, AdamW

import warnings

warnings.filterwarnings('ignore')


def dev_split(dataset_dir):
    data = np.load(dataset_dir, allow_pickle=True)
    words = data["words"]
    labels = data["labels"]
    x_train, x_dev, y_train, y_dev = train_test_split(words, labels, test_size=config.dev_split_size, random_state=0)
    return x_train, x_dev, y_train, y_dev


def test():
    data = np.load(config.test_dir, allow_pickle=True)
    word_test = data["words"]
    label_test = data["labels"]
    test_dataset = NERDataset(word_test, label_test, config)
    logging.info("--------Dataset Build!--------")
    test_loader = DataLoader(test_dataset, batch_size=config.batch_size,
                             shuffle=False, collate_fn=test_dataset.collate_fn)
    logging.info("--------Get Data-loader!--------")
    if config.model_dir is not None:
        model = BertNER.from_pretrained(config.model_dir)
        model.to(config.device)
        logging.info("--------Load model from {}--------".format(config.model_dir))
    else:
        logging.info("--------No model to test !--------")
        return
    val_metrics = evaluate(test_loader, model, mode='test')
    val_f1 = val_metrics['f1']
    logging.info("test loss: {}, f1 score: {}".format(val_metrics['loss'], val_f1))
    val_f1_labels = val_metrics['f1_labels']
    for label in config.labels:
        logging.info("f1 score of {}: {}".format(label, val_f1_labels[label]))

def load_dev(mode):
    if mode == 'train':
        word_train, word_dev, label_train, label_dev = dev_split(config.train_dir)
    elif mode == 'test':
        train_data = np.load(config.train_dir, allow_pickle=True)
        dev_data = np.load(config.test_dir, allow_pickle=True)
        word_train = train_data["words"]
        label_train = train_data["labels"]
        word_dev = dev_data["words"]
        label_dev = dev_data["labels"]
    else:
        word_train = None
        label_train = None
        word_dev = None
        label_dev = None
    return word_train, word_dev, label_train, label_dev


def run():
    utils.set_logger(config.log_dir)
    logging.info("device: {}".format(config.device))
    processor = Processor(config)
    processor.process()
    logging.info("--------Process Done!--------")
    word_train, word_dev, label_train, label_dev = load_dev('train')
    train_dataset = NERDataset(word_train, label_train, config)
    dev_dataset = NERDataset(word_dev, label_dev, config)
    logging.info("--------Dataset Build!--------")
    train_size = len(train_dataset)
    train_loader = DataLoader(train_dataset, batch_size=config.batch_size,
                              shuffle=True, collate_fn=train_dataset.collate_fn)
    dev_loader = DataLoader(dev_dataset, batch_size=config.batch_size,
                              shuffle=True, collate_fn=dev_dataset.collate_fn)
    logging.info("--------Get Dataloader!--------")
    device = config.device
    model = BertNER.from_pretrained(config.roberta_model, num_labels=len(config.label2id))
    model.to(device)
    if config.full_fine_tuning:
        bert_optimizer = list(model.bert.named_parameters())
        lstm_optimizer = list(model.bilstm.named_parameters())
        classifier_optimizer = list(model.classifier.named_parameters())
        no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in bert_optimizer if not any(nd in n for nd in no_decay)],
             'weight_decay': config.weight_decay},
            {'params': [p for n, p in bert_optimizer if any(nd in n for nd in no_decay)],
             'weight_decay': 0.0},
            {'params': [p for n, p in lstm_optimizer if not any(nd in n for nd in no_decay)],
             'lr': config.learning_rate * 5, 'weight_decay': config.weight_decay},
            {'params': [p for n, p in lstm_optimizer if any(nd in n for nd in no_decay)],
             'lr': config.learning_rate * 5, 'weight_decay': 0.0},
            {'params': [p for n, p in classifier_optimizer if not any(nd in n for nd in no_decay)],
             'lr': config.learning_rate * 5, 'weight_decay': config.weight_decay},
            {'params': [p for n, p in classifier_optimizer if any(nd in n for nd in no_decay)],
             'lr': config.learning_rate * 5, 'weight_decay': 0.0},
            {'params': model.crf.parameters(), 'lr': config.learning_rate * 5}
        ]
    else:
        param_optimizer = list(model.classifier.named_parameters())
        optimizer_grouped_parameters = [{'params': [p for n, p in param_optimizer]}]
    optimizer = AdamW(optimizer_grouped_parameters, lr=config.learning_rate, correct_bias=False)
    train_steps_per_epoch = train_size // config.batch_size
    scheduler = get_cosine_schedule_with_warmup(optimizer,
                                                num_warmup_steps=(config.epoch_num // 10) * train_steps_per_epoch,
                                                num_training_steps=config.epoch_num * train_steps_per_epoch)

    logging.info("--------Start Training!--------")
    train(train_loader, dev_loader, model, optimizer, scheduler, config.model_dir)


if __name__ == '__main__':
    run()
    test()


"""
/data0/envs/diveintodl/bin/python3 /data0/shouxieai/ner/hemingkx-bert-lstm-crf/run.py 
device: cuda:0
--------Process Done!--------
--------Dataset Build!--------
--------Get Dataloader!--------
Some weights of the model checkpoint at pretrained_bert_models/chinese_roberta_wwm_large_ext/ were not used when initializing BertNER: ['cls.predictions.transform.dense.bias', 'cls.predictions.bias', 'cls.seq_relationship.weight', 'cls.predictions.transform.LayerNorm.bias', 'cls.seq_relationship.bias', 'cls.predictions.transform.dense.weight', 'cls.predictions.transform.LayerNorm.weight', 'cls.predictions.decoder.weight']
- This IS expected if you are initializing BertNER from the checkpoint of a model trained on another task or with another architecture (e.g. initializing a BertForSequenceClassification model from a BertForPreTraining model).
- This IS NOT expected if you are initializing BertNER from the checkpoint of a model that you expect to be exactly identical (initializing a BertForSequenceClassification model from a BertForSequenceClassification model).
Some weights of BertNER were not initialized from the model checkpoint at pretrained_bert_models/chinese_roberta_wwm_large_ext/ and are newly initialized: ['bilstm.weight_hh_l0_reverse', 'classifier.weight', 'bilstm.bias_hh_l1_reverse', 'bilstm.bias_ih_l1', 'bilstm.weight_hh_l1_reverse', 'bilstm.weight_hh_l1', 'classifier.bias', 'bilstm.weight_ih_l0_reverse', 'bilstm.bias_ih_l0_reverse', 'bilstm.bias_hh_l0', 'bilstm.weight_hh_l0', 'bilstm.bias_ih_l1_reverse', 'bilstm.weight_ih_l0', 'bilstm.weight_ih_l1_reverse', 'crf.start_transitions', 'crf.transitions', 'bilstm.weight_ih_l1', 'bilstm.bias_ih_l0', 'bilstm.bias_hh_l0_reverse', 'crf.end_transitions', 'bilstm.bias_hh_l1']
You should probably TRAIN this model on a down-stream task to be able to use it for predictions and inference.
--------Start Training!--------
100%|██████████| 303/303 [03:12<00:00,  1.57it/s]
Epoch: 1, train loss: 1301.467333299492
Epoch: 1, dev loss: 386.8523501227884, f1 score: 0.6296223830952846
--------Save best model!--------
100%|██████████| 303/303 [03:13<00:00,  1.57it/s]
Epoch: 2, train loss: 304.71115323812654
Epoch: 2, dev loss: 288.2237149406882, f1 score: 0.7221783741120757
--------Save best model!--------
100%|██████████| 303/303 [03:13<00:00,  1.56it/s]
Epoch: 3, train loss: 194.30160567784074
Epoch: 3, dev loss: 269.81662525850186, f1 score: 0.7617733990147784
--------Save best model!--------
100%|██████████| 303/303 [03:13<00:00,  1.56it/s]
Epoch: 4, train loss: 144.3301034304175
Epoch: 4, dev loss: 281.52301833208867, f1 score: 0.7544413738649822
100%|██████████| 303/303 [03:14<00:00,  1.56it/s]
Epoch: 5, train loss: 114.07680882872528
Epoch: 5, dev loss: 291.3105322893928, f1 score: 0.753920619554695
100%|██████████| 303/303 [03:14<00:00,  1.56it/s]
Epoch: 6, train loss: 87.15761981624188
Epoch: 6, dev loss: 299.1374184103573, f1 score: 0.7769871925188045
--------Save best model!--------
100%|██████████| 303/303 [03:14<00:00,  1.56it/s]
Epoch: 7, train loss: 67.28778796935632
Epoch: 7, dev loss: 318.70466748405903, f1 score: 0.7740374924410401
100%|██████████| 303/303 [03:14<00:00,  1.56it/s]
Epoch: 8, train loss: 50.252787951982455
Epoch: 8, dev loss: 353.34314054601333, f1 score: 0.7652457678972057
100%|██████████| 303/303 [03:14<00:00,  1.56it/s]
Epoch: 9, train loss: 42.37378379141931
Epoch: 9, dev loss: 400.34927457921646, f1 score: 0.7738652554447384
100%|██████████| 303/303 [03:14<00:00,  1.56it/s]
Epoch: 10, train loss: 35.08923584714581
Epoch: 10, dev loss: 384.0724639892578, f1 score: 0.7806661251015434
--------Save best model!--------
100%|██████████| 303/303 [03:14<00:00,  1.56it/s]
Epoch: 11, train loss: 28.617883738904897
Epoch: 11, dev loss: 433.9839432660271, f1 score: 0.7757160145219847
100%|██████████| 303/303 [03:14<00:00,  1.56it/s]
Epoch: 12, train loss: 26.2786072079498
Epoch: 12, dev loss: 426.5895125445198, f1 score: 0.7763475462590508
100%|██████████| 303/303 [03:14<00:00,  1.56it/s]
Epoch: 13, train loss: 23.66120773416148
Epoch: 13, dev loss: 427.90251877728633, f1 score: 0.7725147387680422
100%|██████████| 303/303 [03:14<00:00,  1.56it/s]
Epoch: 14, train loss: 19.327330560967475
Epoch: 14, dev loss: 433.20255279541016, f1 score: 0.7736228387615601
100%|██████████| 303/303 [03:14<00:00,  1.56it/s]
Epoch: 15, train loss: 18.55907702052554
Epoch: 15, dev loss: 532.7707901000977, f1 score: 0.7835051546391751
--------Save best model!--------
100%|██████████| 303/303 [03:14<00:00,  1.56it/s]
Epoch: 16, train loss: 16.117935394689983
Epoch: 16, dev loss: 575.4862774119657, f1 score: 0.7774625050668829
100%|██████████| 303/303 [03:14<00:00,  1.56it/s]
Epoch: 17, train loss: 17.44241469606708
Epoch: 17, dev loss: 561.6833119111902, f1 score: 0.7767263942017314
100%|██████████| 303/303 [03:14<00:00,  1.56it/s]
Epoch: 18, train loss: 16.11264682052159
Epoch: 18, dev loss: 546.3473780014936, f1 score: 0.7749345154140641
100%|██████████| 303/303 [03:14<00:00,  1.56it/s]
Epoch: 19, train loss: 13.198873362525461
Epoch: 19, dev loss: 597.4523997587316, f1 score: 0.7766239055182244
100%|██████████| 303/303 [03:14<00:00,  1.56it/s]
Epoch: 20, train loss: 12.191983685635105
Epoch: 20, dev loss: 637.1317699656767, f1 score: 0.7912087912087913
--------Save best model!--------
100%|██████████| 303/303 [03:14<00:00,  1.56it/s]
Epoch: 21, train loss: 10.668551101936366
Epoch: 21, dev loss: 715.6174675436581, f1 score: 0.7793522267206479
100%|██████████| 303/303 [03:14<00:00,  1.56it/s]
Epoch: 22, train loss: 12.26239831376784
Epoch: 22, dev loss: 717.5901327694164, f1 score: 0.7817707287693235
100%|██████████| 303/303 [03:14<00:00,  1.56it/s]
Epoch: 23, train loss: 9.75768000536626
Epoch: 23, dev loss: 719.4064124612247, f1 score: 0.7793522267206479
100%|██████████| 303/303 [03:13<00:00,  1.56it/s]
Epoch: 24, train loss: 8.722719016248243
Epoch: 24, dev loss: 729.7780784158145, f1 score: 0.7815906338312475
100%|██████████| 303/303 [03:14<00:00,  1.56it/s]
Epoch: 25, train loss: 7.601173898174425
Epoch: 25, dev loss: 745.7835415110868, f1 score: 0.7837457252061959
100%|██████████| 303/303 [03:14<00:00,  1.56it/s]
Epoch: 26, train loss: 8.011492279103093
Epoch: 26, dev loss: 774.7719116210938, f1 score: 0.7819702223128696
100%|██████████| 303/303 [03:14<00:00,  1.56it/s]
Epoch: 27, train loss: 6.064915213254419
Epoch: 27, dev loss: 759.8499293607824, f1 score: 0.7878542510121457
100%|██████████| 303/303 [03:14<00:00,  1.56it/s]
Epoch: 28, train loss: 8.3529054937583
Epoch: 28, dev loss: 749.184637630687, f1 score: 0.7932161830813241
--------Save best model!--------
100%|██████████| 303/303 [03:14<00:00,  1.56it/s]
Epoch: 29, train loss: 4.386884053548177
Epoch: 29, dev loss: 765.1020651424633, f1 score: 0.7867855695176329
100%|██████████| 303/303 [03:14<00:00,  1.56it/s]
Epoch: 30, train loss: 4.6652951759867145
Epoch: 30, dev loss: 804.6529505112592, f1 score: 0.7828907358605312
100%|██████████| 303/303 [03:14<00:00,  1.56it/s]
Epoch: 31, train loss: 1.9802815670227454
Epoch: 31, dev loss: 815.3560674330768, f1 score: 0.7829923786602486
100%|██████████| 303/303 [03:14<00:00,  1.56it/s]
Epoch: 32, train loss: 2.3535953677526793
Epoch: 32, dev loss: 795.5591686473173, f1 score: 0.7854678303227116
100%|██████████| 303/303 [03:14<00:00,  1.56it/s]
Epoch: 33, train loss: 2.8136963450869317
Epoch: 33, dev loss: 834.9371364817899, f1 score: 0.7863940068839845
100%|██████████| 303/303 [03:14<00:00,  1.56it/s]
Epoch: 34, train loss: 1.9998819625416997
Epoch: 34, dev loss: 825.2955977495978, f1 score: 0.7903752039151712
100%|██████████| 303/303 [03:14<00:00,  1.56it/s]
Epoch: 35, train loss: 0.7073552309483191
Epoch: 35, dev loss: 826.933677224552, f1 score: 0.7919708029197081
100%|██████████| 303/303 [03:14<00:00,  1.56it/s]
Epoch: 36, train loss: 1.2457344764136637
Epoch: 36, dev loss: 859.5019208122702, f1 score: 0.7893991503135748
100%|██████████| 303/303 [03:13<00:00,  1.56it/s]
Epoch: 37, train loss: 0.7027597458842564
Epoch: 37, dev loss: 836.8060437370749, f1 score: 0.787952787952788
100%|██████████| 303/303 [03:13<00:00,  1.56it/s]
Epoch: 38, train loss: 0.6759850160516921
Epoch: 38, dev loss: 840.538652307847, f1 score: 0.7925370107483269
Best val f1: 0.7932161830813241
Training Finished!
--------Dataset Build!--------
--------Get Data-loader!--------
--------Load model from /data0/shouxieai/ner/hemingkx-bert-lstm-crf/experiments/clue/--------
--------Bad Cases reserved !--------
test loss: 794.619883219401, f1 score: 0.7884179877062439
f1 score of address: 0.6357615894039735
f1 score of book: 0.8157894736842106
f1 score of company: 0.8104575163398693
f1 score of game: 0.8366013071895424
f1 score of government: 0.7976878612716763
f1 score of movie: 0.7931034482758621
f1 score of name: 0.8862911795961742
f1 score of organization: 0.7848443843031123
f1 score of position: 0.784452296819788
f1 score of scene: 0.7107843137254902

Process finished with exit code 0

"""