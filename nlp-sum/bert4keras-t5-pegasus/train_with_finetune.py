import os
import re
import rouge
import jieba
import torch
import argparse
import numpy as np
from tqdm.auto import tqdm
from bert4torch.models import *
from torch.utils.data import DataLoader, Dataset
from torch._six import container_abcs, string_classes, int_classes
from transformers import MT5ForConditionalGeneration, BertTokenizer


def load_data(filename):
    """加载数据
    单条格式：(标题, 正文)
    """
    D = []
    with open(filename, encoding='utf-8') as f:
        for l in f.readlines():
            cur = l.strip().split('\t')
            if len(cur) == 2:
                title, content = cur[0], cur[1]
                D.append((title, content))
            elif len(cur) == 1:
                content = cur[0]
                D.append(content)
    return D


class T5PegasusTokenizer(BertTokenizer):
    """结合中文特点完善的Tokenizer
    基于词颗粒度的分词，如词表中未出现，再调用BERT原生Tokenizer
    """
    def __init__(self, pre_tokenizer=lambda x: jieba.cut(x, HMM=False), *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.pre_tokenizer = pre_tokenizer

    def _tokenize(self, text, *arg, **kwargs):
        split_tokens = []
        for text in self.pre_tokenizer(text):
            if text in self.vocab:
                split_tokens.append(text)
            else:
                split_tokens.extend(super()._tokenize(text))
        return split_tokens


class KeyDataset(Dataset):
    def __init__(self, dict_data):
        self.data = dict_data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index]


def create_data(data, tokenizer, max_len=512, term='train'):
    """调用tokenizer.encode编码正文/标题，每条样本用dict表示数据域
    """
    ret, flag = [], True
    for title, content in data:
        text_ids = tokenizer.encode(content, max_length=max_len, truncation='only_first')
        if flag and term == 'train':
            flag = False
            print(content)
        if term == 'train':
            summary_ids = tokenizer.encode(title, max_length=max_len, truncation='only_first')
            features = {'input_ids': text_ids,
                        'decoder_input_ids': summary_ids,
                        'attention_mask': [1] * len(text_ids),
                        'decoder_attention_mask': [1] * len(summary_ids)
                       }
        
        elif term == 'dev':
            features = {'input_ids': text_ids,
                        'attention_mask': [1] * len(text_ids),
                        'title': title
                       }
            
        ret.append(features)
    return ret


def sequence_padding(inputs, length=None, padding=0):
    """Numpy函数，将序列padding到同一长度
    """
    if length is None:
        length = max([len(x) for x in inputs])

    pad_width = [(0, 0) for _ in np.shape(inputs[0])]
    outputs = []
    for x in inputs:
        x = x[:length]
        pad_width[0] = (0, length - len(x))
        x = np.pad(x, pad_width, 'constant', constant_values=padding)
        outputs.append(x)

    return np.array(outputs, dtype='int64')


def default_collate(batch):
    """组batch
    各个数据域分别转换为tensor，tensor第一个维度等于batch_size
    """
    np_str_obj_array_pattern = re.compile(r'[SaUO]')
    default_collate_err_msg_format = (
        "default_collate: batch must contain tensors, numpy arrays, numbers, "
        "dicts or lists; found {}")
    elem = batch[0]
    elem_type = type(elem)
    if isinstance(elem, torch.Tensor):
        out = None
        if torch.utils.data.get_worker_info() is not None:
            # If we're in a background process, concatenate directly into a
            # shared memory tensor to avoid an extra copy
            numel = sum([x.numel() for x in batch])
            storage = elem.storage()._new_shared(numel)
            out = elem.new(storage)
        return torch.stack(batch, 0, out=out)
    elif elem_type.__module__ == 'numpy' and elem_type.__name__ != 'str_' \
            and elem_type.__name__ != 'string_':
        if elem_type.__name__ == 'ndarray' or elem_type.__name__ == 'memmap':
            # array of string classes and object
            if np_str_obj_array_pattern.search(elem.dtype.str) is not None:
                raise TypeError(default_collate_err_msg_format.format(elem.dtype))
            return default_collate([torch.as_tensor(b) for b in batch])
        elif elem.shape == ():  # scalars
            return torch.as_tensor(batch)
    elif isinstance(elem, float):
        return torch.tensor(batch, dtype=torch.float64)
    elif isinstance(elem, int_classes):
        return torch.tensor(batch, dtype=torch.long)
    elif isinstance(elem, string_classes):
        return batch
    elif isinstance(elem, container_abcs.Mapping):
        return {key: default_collate([d[key] for d in batch]) for key in elem}
    elif isinstance(elem, tuple) and hasattr(elem, '_fields'):  # namedtuple
        return elem_type(*(default_collate(samples) for samples in zip(*batch)))
    elif isinstance(elem, container_abcs.Sequence):
        # check to make sure that the elements in batch have consistent size
        it = iter(batch)
        elem_size = len(next(it))
        if not all(len(elem) == elem_size for elem in it):
            batch = sequence_padding(batch)
        return default_collate([default_collate(elem) for elem in batch])
    raise TypeError(default_collate_err_msg_format.format(elem_type))


def prepare_data(args, data_path, tokenizer, term='train'):
    """准备batch数据
    """
    data = load_data(data_path)
    data = create_data(data, tokenizer, args.max_len, term)
    data = KeyDataset(data)
    data = DataLoader(data, batch_size=args.batch_size, collate_fn=default_collate)
    return data


def compute_rouge(source, target):
    """计算rouge-1、rouge-2、rouge-l
    """
    source, target = ' '.join(source), ' '.join(target)
    try:
        scores = rouge.Rouge().get_scores(hyps=source, refs=target)
        return {
            'rouge-1': scores[0]['rouge-1']['f'],
            'rouge-2': scores[0]['rouge-2']['f'],
            'rouge-l': scores[0]['rouge-l']['f'],
        }
    except ValueError:
        return {
            'rouge-1': 0.0,
            'rouge-2': 0.0,
            'rouge-l': 0.0,
        }
    
    
def compute_rouges(sources, targets):
    scores = {
        'rouge-1': 0.0,
        'rouge-2': 0.0,
        'rouge-l': 0.0,
    }
    for source, target in zip(sources, targets):
        score = compute_rouge(source, target)
        for k, v in scores.items():
            scores[k] = v + score[k]

    return {k: v / len(targets) for k, v in scores.items()}


def train_model(model, adam, train_data, dev_data, tokenizer, device, args):
    if not os.path.exists(args.model_dir):
        os.mkdir(args.model_dir)
        
    best = 0
    for epoch in range(args.num_epoch):
        model.train()
        for i, cur in enumerate(tqdm(train_data, desc='Epoch {}:'.format(epoch))):
            cur = {k: v.to(device) for k, v in cur.items()}
            prob = model(**cur)[0]
            mask = cur['decoder_attention_mask'][:, 1:].reshape(-1).bool()
            prob = prob[:, :-1]
            prob = prob.reshape((-1, prob.size(-1)))[mask]
            labels = cur['decoder_input_ids'][:, 1:].reshape(-1)[mask]
            loss_fct = torch.nn.CrossEntropyLoss(ignore_index=-100)
            loss = loss_fct(prob, labels)
            if i % 100 == 0:
                print("Iter {}:  Training Loss: {}".format(i, loss.item()))
            loss.backward()
            adam.step()
            adam.zero_grad()

        # 验证
        model.eval()
        gens = []
        summaries = []
        for feature in tqdm(dev_data):
            title = feature['title']
            content = {k : v.to(device) for k, v in feature.items() if k != 'title'} 
            if args.data_parallel and torch.cuda.is_available():
                gen = model.module.generate(max_length=args.max_len_generate,
                             eos_token_id=tokenizer.sep_token_id,
                             decoder_start_token_id=tokenizer.cls_token_id,
                             **content)
            else:
                gen = model.generate(max_length=args.max_len_generate,
                             eos_token_id=tokenizer.sep_token_id,
                             decoder_start_token_id=tokenizer.cls_token_id,
                             **content)
            gen = tokenizer.batch_decode(gen, skip_special_tokens=True)
            gen = [item.replace(' ', '') for item in gen]
            # print(title)
            # print(gen)
            gens.extend(gen)
            summaries.extend(title)
        scores = compute_rouges(gens, summaries)
        print("Validation Loss: {}".format(scores))
        rouge_l = scores['rouge-l']
        if rouge_l > best:
            best = rouge_l
            if args.data_parallel and torch.cuda.is_available():
                torch.save(model.module, os.path.join(args.model_dir, 'summary_model'))
            else:
                torch.save(model, os.path.join(args.model_dir, 'summary_model'))
        # torch.save(model, os.path.join(args.model_dir, 'summary_model_epoch_{}'.format(str(epoch))))


def init_argument():
    parser = argparse.ArgumentParser(description='t5-pegasus-chinese')
    parser.add_argument('--train_data', default='./data/train.tsv')
    parser.add_argument('--dev_data', default='./data/dev.tsv')
    parser.add_argument('--pretrain_model', default='./t5_pegasus_pretrain')
    parser.add_argument('--model_dir', default='./saved_model')
    
    parser.add_argument('--num_epoch', default=20, help='number of epoch')
    #parser.add_argument('--batch_size', default=16, help='batch size')
    parser.add_argument('--batch_size', default=8, help='batch size')
    parser.add_argument('--lr', default=2e-4, help='learning rate')
    parser.add_argument('--data_parallel', default=False)
    #parser.add_argument('--max_len', default=512, help='max length of inputs')
    parser.add_argument('--max_len', default=1024, help='max length of inputs')
    #parser.add_argument('--max_len_generate', default=40, help='max length of outputs')
    parser.add_argument('--max_len_generate', default=64, help='max length of outputs')

    args = parser.parse_args()
    return args

if __name__ == '__main__':

    # step 1. init argument
    args = init_argument()

    # step 2. prepare training data and validation data
    tokenizer = T5PegasusTokenizer.from_pretrained(args.pretrain_model)
    train_data = prepare_data(args, args.train_data, tokenizer, term='train')
    dev_data = prepare_data(args, args.dev_data, tokenizer, term='dev')

    # step 3. load pretrain model
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = MT5ForConditionalGeneration \
                .from_pretrained(args.pretrain_model).to(device)
    if args.data_parallel and torch.cuda.is_available():
        device_ids = range(torch.cuda.device_count())
        model = torch.nn.DataParallel(model, device_ids=device_ids)
    
    # step 4. finetune
    adam = torch.optim.Adam(model.parameters(), lr=args.lr)
    train_model(model, adam, train_data, dev_data, tokenizer, device, args)

'''
pip3 install torch==1.7.1+cu110  -f https://download.pytorch.org/whl/cu110/torch_stable.html

    #parser.add_argument('--batch_size', default=16, help='batch size')
    parser.add_argument('--batch_size', default=32, help='batch size')
    parser.add_argument('--lr', default=2e-4, help='learning rate')
    parser.add_argument('--data_parallel', default=False)
    #parser.add_argument('--max_len', default=512, help='max length of inputs')
    parser.add_argument('--max_len', default=1024, help='max length of inputs')
    #parser.add_argument('--max_len_generate', default=40, help='max length of outputs')
    parser.add_argument('--max_len_generate', default=64, help='max length of outputs')
small版本效果不太好：
而消息人士称,章子怡原来打算在演唱会上当着章子怡的面宣布重大消息,而且章子怡已经赴上海准备参加演唱会了    
     四海网讯,近日,有媒体报道称:章子怡真怀孕了!大概四五个月,预产期是年底前后,现在已经不接工作了。”这到底是怎么回事?消息是真是假?针对此消息,23日晚8时30分,华西都市报记者迅速联系上了与章子怡家里关亲十分高兴。子怡的母亲,已开始悉心照料女儿了。子怡的预产期大概是今年12月底。”当晚9时,华西都市报记者为了求证章子怡怀孕消息,又电话联系章子怡的亲哥哥章传N遍了!不过,时间跨入2015年,事情却发生着微妙的变化。2015年3月21日,章子怡担任制片人的电影《从天儿降》开机,在开机发布会上几张合影,让网友又燃起了好奇怡又被发现状态不佳,不时深呼吸,不自觉想捂住肚子,又觉得不妥。然后在8月的一天,章子怡和朋友吃饭,在酒店门口被风行工作室拍到了,疑似有孕在身!今年7月11日,赴上海准备参加演唱会了,怎知遇到台风,只好延期,相信9月26日的演唱会应该还会有惊喜大白天下吧。
根据世界自然保护联盟和中国野生动植物和自然保护区管理局的相关专家对其进行了详尽的记录,但由此也有人判断为从动物园逃亡
     中新社西宁11月22日电(赵凛松植物和自然保护区管理局高级工程师张毓22日向中新社记者确认:“经过中国林业科学院、中科院新疆生态与地理研究所和青海省林业厅的共同认定,出现在青海省海西州站在野外监测巡护过程中,在可鲁克湖西南岸入水口盐沼滩发现三只体型较大的鸟类。张毓说:“此前在该区域从未发现过这种体型的鸟类。”可鲁克湖—托素湖位于青海省论,然后会同中国林业科学院和中科院新疆生态与地理研究所的相关专家,确认了这三只鸟为红鹳目红鹳科红鹳属的大红鹳。大红鹳也称为大火烈鸟、红鹤等,三只鸟类特数量较大,无威胁因子,以往在中国并无分布。但1997年在新疆野外首次发现并确定该鸟在中国境内有分布,为中国鸟类新纪录,2012年在四川也发现一只该鸟亚成体。此饲养,因此也有人判断为从动物园逃逸。“我们对这三只鸟进行了详尽的记录,如果明年这个时间还在此地出现这种鸟,那就能肯定是迁徙的鸟类,而不是从动物园里跑出来结冰,鸟类采食困难,不排除三只鸟由于无法获得能量补给而进行远距离迁飞的可能。青海省林业厅野生动物行政主管部门将随时做好野外救护的各项准备工作。
就这样,杨丽有了自己的女儿,孩子出生后,杨丽一直待在家中照顾小孩工作家务,而杨丽却只能待在家中带小孩,规规   
     内容提要:因为早早结婚,今年20岁的杨丽(化名)31日,和老公大吵一架的她离家出走,直到11月12日才回到租住在东莞大朗的家,提出不想在家带孩子要出去打工被父亲拒绝后,杨丽竟欲跳楼轻生,幸好被消防官兵救下。今年10月31日,和老公大吵一架的她离家出走,直到11月12日才回到租住在东莞大朗的家,提出不想在家带孩子要出去打工被父亲拒绝后,杨丽竟欲跳楼轻生,幸好被消防官不知去向。“我还年轻,能打工养活她。”杨丽的父亲告诉羊城晚报记者,他希望女儿能想通,回到父母身边。不愿年少就做主妇在重庆老家读到初二,16岁的杨丽就辍学来17岁的杨丽有了自己的女儿,孩子出生后,杨丽一直待在家中照顾小孩打理家务,而丈夫则在惠州从事手机销售的工作,杨丽就跟父母住在一起。杨父称,在女儿照顾小孩的,规规矩矩做一名家庭主妇。杨父听后,觉得应该让女儿出去走走看看,见见“外面的世界”。杨父便辞去了自己的工作,回来照顾外孙女,女儿则进入大朗一家工厂上班。做后,女儿离家出走了一段时间,直到11月12日才回来,询问后才知道女儿与女婿吵架了,但为何吵架,老杨也不知道。被救后携女离去本月13日,杨丽再次提出不想带小孩想跳楼的正是自己的女儿。随后,他和家人苦口婆心劝说女儿不要做傻事。当天,大朗消防中队接到女子要跳楼的报警电话。消防官兵赶到后,发现女子坐在一栋出租屋四楼后,在隔壁房间的窗户对女子进行了劝说,一番劝说无效后,消防员决定带领官兵强行破门而入。当他来到房门口时,听见了小孩在门后的哭叫声,于是考虑使用云梯进行营人应门。电话中,老杨告诉记者,杨丽带着女儿离开了大朗,不知去向。老杨坦言,他不希望女儿出去打工,“她年纪还小,吃不起苦。她能在家中照顾好小孩就是对我们最大的帮助。我还年轻,能打工养活她。”老杨说,他希望女儿能想通,早点回到父母身边。

    parser.add_argument('--num_epoch', default=20, help='number of epoch')
    #parser.add_argument('--batch_size', default=16, help='batch size')
    parser.add_argument('--batch_size', default=16, help='batch size')
    parser.add_argument('--lr', default=2e-4, help='learning rate')
    parser.add_argument('--data_parallel', default=False)
    #parser.add_argument('--max_len', default=512, help='max length of inputs')
    parser.add_argument('--max_len', default=1024, help='max length of inputs')
    #parser.add_argument('--max_len_generate', default=40, help='max length of outputs')
    parser.add_argument('--max_len_generate', default=64, help='max length of outputs')
base版本
而消息人士称，章子怡原来打算在演唱会上张艺谋导演的电影叫天儿降  
     四海网讯,近日,有媒体报道称:章子怡真怀孕了!报道还援引知情人士消息称,“章子怡怀孕大概是假?针对此消息,23日晚8时30分,华西都市报记者迅速联系上了与章子怡家里关系极好的知情人士,这位人士向华西都市报记者证实说:“子怡这次确实怀孕了。她已经3当晚9时,华西都市报记者为了求证章子怡怀孕消息,又电话联系章子怡的亲哥哥章子男,但电话通了,一直没有人接听。有关章子怡怀孕的新闻自从2013年9月份章子怡和微妙的变化。2015年3月21日,章子怡担任制片人的电影《从天儿降》开机,在开机发布会上几张合影,让网友又燃起了好奇心:“章子怡真的怀孕了吗?”但后据证实,章子怡,又觉得不妥。然后在8月的一天,章子怡和朋友吃饭,在酒店门口被风行工作室拍到了,疑似有孕在身!今年7月11日,汪峰本来在上海要举行演唱会,后来因为台风“灿鸿”取相信9月26日的演唱会应该还会有惊喜大白天下吧。
青海湖是青海湖摄氏度的驯摩亚成体
     中新社西宁11月22日电(赵凛松)青海省林业厅野生动植物和自然保护区管理局高级工程师张毓22日向中新社记者确认:“经与地理研究所和青海省林业厅的共同认定,出现在青海省海西州境内的三只体型较大的鸟为世界极度濒危的红鹳目红鹳科红鹳属的大红鹳。”11月18日,青海省海西州可鲁域从未发现过这种体型的鸟类。”可鲁克湖—托素湖位于青海省柴达木盆地东北部,海拔2800米,水域湿地环境内的优势种动物主要是水禽,共有30余种。根据拍摄的照片以鹳属的大红鹳。大红鹳也称为大火烈鸟、红鹤等,三只鸟类特征为大红鹳亚成体。根据世界自然保护联盟、世界濒危动物红色名录,该鸟主要分布于非洲、中亚、南亚等现并确定该鸟在中国境内有分布,为中国鸟类新纪录,2012年在四川也发现一只该鸟亚成体。此次野外发现在中国属第三次。“我们现在还无法判断这三只鸟从何而来。不如果明年这个时间还在此地出现这种鸟,那就能肯定是迁徙的鸟类,而不是从动物园里跑出来的。”由于目前可鲁克湖—托素湖已开始结冰,鸟类采食困难,不排除三只鸟由护的各项准备工作。
就这样,杨父便辞去了自己的女儿
     内容提要:因为早早结婚,今年20岁的杨丽(化名)已经是一个三岁小孩的妈妈。今年10月31日,和老公大吵一架的她离家出走,直到11消防官兵救下。女子欲跳楼轻生视频截图因为早早结婚,今年20岁的杨丽(化名)已经是一个三岁小孩的妈妈。还未好好享受少女时代,她就开始围着孩子、老公转。今年生,幸好被消防官兵救下,14日,她又带着孩子离开了大朗,不知去向。“我还年轻,能打工养活她。”杨丽的父亲告诉羊城晚报记者,他希望女儿能想通,回到父母身边。不愿庆老家读到初二,16岁的杨丽就辍学来到东莞打工。在工厂上班时,她认识了现在的老公,两人很快走到了一起,并有了孩子。杨父原本不同意两人在一起,但女儿肚子大了母住在一起。杨父称,在女儿照顾小孩的两年多时间里,女儿从未跟他们提过要出去打工。但是就在两月前的某天,女儿告诉他,跟她同龄的女孩子都在工厂里打工赚钱,而孙女,女儿则进入大朗一家工厂上班。做到10月底,女儿便辞工不干了。老杨不知道女儿辞工的原因,但他猜测是吃不起苦的原因。辞工后,女儿离家出走了一段时间,直到才回来,询问后才知道女儿与女婿吵架了,但为何吵架,老杨也不知道。被救后携女离去本月13日,杨丽再次提出不想带小孩想去打工。但这次,老杨拒绝了女儿的要求,女劝说女儿不要做傻事。当天,大朗消防中队接到女子要跳楼的报警电话。消防官兵赶到后,发现女子坐在一栋出租屋四楼的窗户上,边哭边喊:“我不想过这样的生活。”据到房门口时,听见了小孩在门后的哭叫声,于是考虑使用云梯进行营救,但杨丽情绪激动,消防官兵只得和家人继续劝说。经过努力,终于进入房内将正准备往下跳的杨丽救许久也无人应门。电话中,老杨告诉记者,杨丽带着女儿离开了大朗,不知去向。老杨坦言,他不希望女儿出去打工,“她年纪还小,吃不起苦。她能在家中照顾好小孩就是对
'''