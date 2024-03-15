from paddlenlp import Taskflow
summarizer = Taskflow("text_summarization")
texts = ['63岁退休教师谢淑华，拉着人力板车，历时1年，走了2万4千里路，带着年过九旬的妈妈环游中国，完成了妈妈“一辈子在锅台边转，也想出去走走”的心愿。她说：“妈妈愿意出去走走，我就愿意拉着，孝心不能等，能走多远就走多远。',
        '2022年，中国房地产进入转型阵痛期，传统“高杠杆、快周转”的模式难以为继，万科甚至直接喊话，中国房地产进入“黑铁时代”。',
        '据悉，2022年教育部将围绕“巩固提高、深化落实、创新突破”三个关键词展开工作。要进一步强化学校教育主阵地作用，继续把落实“双减”作为学校工作的重中之重，重点从提高作业设计水平、提高课后服务水平、提高课堂教学水平、提高均衡发展水平四个方面持续巩固提高学校“双减”工作水平。',
        '其次，《唐朝诡事录》每个单元的故事，都可独立成为一部“电影”观看，且质感极佳。这一点，“甘棠驿”、“黄梅杀”、“石桥图”三个单元尤其明显，不管从故事的布局 还是“氛围感”来说，虽说不上特别惊艳，但是却让 人能看见该剧的满满诚意。',
        '综述了人工神经网络FPGA实现的研究进展和关键技术，分析了如何利用FPGA的可重构技术来实现人工神经网络，探讨了实现过程中的一些问题，并介绍了作为神经网络FPGA实现的基础一可重构技术。指出测试平台设计、软件工具、FPGA友好学习算法及拓扑结构自适应等方面的研究，是今后研究的热点。',
        '党参有降血脂，降血压的作用，可以彻底消除血液中的垃圾，从而对冠心病以及心血管疾病的患者都有一定的稳定预防工作作用，因此平时口服党参能远离三高的危害。另外党参除了益气养血，降低中枢神经作用，调整消化系统功能，健脾补肺的功能。',
        '肾结石一般在0.6以下的是药物排石,常用的药物如肾石通颗粒,对症药物等.大于0.6以上的是体外冲击波碎石,大于2.0以上的可以手术治疗.建议你到医院的泌尿外科进行仔细检查,查明原因,及时治疗.肾结石一粒直径大于1CM就很难通过药物排出来.可以做碎石治疗,如果高于1CM,可以用中药的排石汤,抗菌素和对症治疗.当前先解决疼痛,稳定情绪,稳定体质.平时需发生改变饮食结构和习惯,建议多做剧烈运动或多饮水。',
        '过户流程为，现场竞买成功后，买受人应立即支付成交价一定比例的款项作为定金，并在拍卖行协助>下与委托人签订拍卖房地产转让合同书。买受人再支付全部价款后，凭转让合同书和相关证明文件到房地产登记机关办理产权过户手续，取得房地产权证。房屋司法拍卖流程一、接受拍卖委托在这一阶段，委托人将有意要拍卖的房产明确委托给拍卖行，双方签订委托房产拍卖协议，对委托拍卖的标的物达成基本意向。委托人在委托时一般要想拍卖行提供房地产权证、身份证等相关房产证明文件。二、拍卖房产标的的调查确认拍卖行对房产委托人提供的房地产产权证明、有关文件、房产证明材料等进一步核实，必要时到相关部门调查取证，同时还必须对房产进行现场勘查。三、接受委托、签订委托、拍卖合同书经调查确认后，拍卖行认为符合拍卖委托条件的，与委托人签订委托拍卖合同。委托拍卖必须符合《拍卖法》的要求。委托拍卖合同中要对拍卖房地产的情况、拍卖费用、拍卖方式和期限、违约责任等加以明确。四、房地产拍卖底价的确定拍卖行对房地产市场进行调查与分析，必要时请专业的房地产估价人员对拍卖房地产进行价格评估，与委托方共同商谈，后确定拍卖底价和起拍价。五、发布拍卖公告，组织接待竞买人拍卖行一般要在拍卖日前半个月至一个月前登报或通过电视媒体以公告形式发布拍卖房地产的信息，拍卖行要对公告的内容真实性负责。同时，组织接待竞买人，向竞买人提供资料，审查竞买人资格，收取保证金，完成竞买人登记。',
        ]
for text in texts:
    #print(f'Content: {text}\n')
    title = summarizer(text)
    print(f'Title: {title[0]}')

'''
六旬老教师1年走2万4千里路带妈妈游中国
万科喊话中国房地产进入“黑铁时代”
教育部：将从四个方面持续巩固提高学校“双减”工作水平
《唐朝诡事录》：不一样的精彩
人工神经网络fpga 实现研究进展和关键技术
党参能降低三高的危害
肾结石相关知识
房屋司法拍卖流程
'''

'''
Title: 年过九旬的“妈妈”在中国“走”了2万4千里
[2024-02-29 22:08:58,755] [ WARNING] - `max_length` will be deprecated in future releases, use `max_new_tokens` instead.
Title: 中国房地产进入“黑铁时代”
[2024-02-29 22:09:00,838] [ WARNING] - `max_length` will be deprecated in future releases, use `max_new_tokens` instead.
Title: 
[2024-02-29 22:09:02,020] [ WARNING] - `max_length` will be deprecated in future releases, use `max_new_tokens` instead.
Title: 2012年10月25日电视之最
[2024-02-29 22:09:04,838] [ WARNING] - `max_length` will be deprecated in future releases, use `max_new_tokens` instead.
Title: 
[2024-02-29 22:09:06,074] [ WARNING] - `max_length` will be deprecated in future releases, use `max_new_tokens` instead.
Title: 有有降血压的作用
[2024-02-29 22:09:08,539] [ WARNING] - `max_length` will be deprecated in future releases, use `max_new_tokens` instead.
Title: 你的小小小年
[2024-02-29 22:09:11,626] [ WARNING] - `max_length` will be deprecated in future releases, use `max_new_tokens` instead.
Title: 如何在网上卖房
'''

'''
summarizer('四海网讯,近日,有媒体报道称:章子怡真怀孕了!报道还援引知情人士消息称,“章子怡怀孕大概四五个月,预产期是年底前后,现在已经不接工作了。” 这到底是怎么回事?消息是真是假?针对此消息,23日晚8时30分,华西都市报记者迅速联系上了与章子怡家里关系极好的知情人士,这位人士向华西都市报记者证实说:“子产期大概是今年12 月底。”当晚9时,华西都市报记者为了求证章子怡怀孕消息,又电话联系章子怡的亲哥哥章子男,但电话通了,一直没有人接听。有关章子怡怀孕的新闻章子怡和汪峰恋情以来,就被传N遍了!不过,时间跨入2015年,事情却发生着微妙的变化。2015年3月21日,章子怡担任制片人的电影《从天儿降》开机,在开机发布会上几好奇心:“章子怡真的怀孕了吗?”但后据证实,章子怡的“大肚照”只是影片宣传的噱头。过了四个月的7月22日,《太平轮》新一轮宣传,章子怡又 被发现状态不佳,不时深子,又觉得不妥。然后在8月的一天,章子怡和朋友吃饭,在酒店门口被风行工作室拍到了,疑似有孕在身!今年7月11日,汪峰本来在上海要举行演唱会,后来因为台风“灿鸿了,怎知遇到台风,只好延期,相信9月26日的演唱会应该还会有惊喜大白天下吧。')
[2024-03-04 16:44:17,974] [ WARNING] - `max_length` will be deprecated in future releases, use `max_new_tokens` instead.
['媒体称已证实子怡已怀孕']现
summarizer('中新社西宁11月22日电(赵凛松)青海省林业厅野生动植物和自然保护区管理局高级工程师张毓22日向中新社记者确认:“经过中国林业科学院、中科院的大红鹳。”11月18日,青海省 海西州可鲁克湖—托素湖国家级陆生野生动物疫源疫病监测站在野外监测巡护过程中,在可鲁克湖西南岸入水口盐沼滩发现三只体型较大的水域湿地环境内的优势种动物主要是水禽,共有30余种。根据拍摄的照片以 及视频,张毓根据动物学体型得出了初步结论,然后会同中国林业科学院和中科院新疆生态与为红鹳目红鹳科红鹳属的大红鹳。 大红鹳也称为大火烈鸟、红鹤等,三只鸟类特征为大红鹳亚成体。根据世界自然保护联盟、世界濒危动物红色名录,该鸟主要分布于非种群数量较大,无威胁因子,以往在中国并无分布。但1997年在新疆野外首次发现并确定该鸟在中国境内有分布,为中国鸟类新纪录,2012年在四川也发现一只该鸟亚成体人工饲养,因此也有人判断为从动物园逃逸。“我们对这三只鸟进行了详尽的记录,如果明年这个时间还在此地出现这种鸟,那就能肯定是迁徙的鸟类,而不是从动物园里跑目前可鲁克湖—托素湖已开始结冰,鸟类采食困难,不排除三只鸟由于无法获得能量补给而进行远距离迁飞的可能。青海省林业厅野生动物行政主管部门将随时做好野外 救护的各项准备工作。')
[2024-03-04 16:45:17,964] [ WARNING] - `max_length` will be deprecated in future releases, use `max_new_tokens` instead.
['中国“大红大紫”在在青海被发']
summarizer('内容提要:因为早早结婚,今年20岁的杨丽(化名)已经是一个三岁小孩的妈妈。今年10月31日,和老公大吵一架的她离家出走,直到11月12日才回到租住截图因为早早结婚,今年20岁的杨丽(化 名)已经是一个三岁小孩的妈妈。还未好好享受少女时代,她就开始围着孩子、老公转。今年10月31日,和老公大吵一架的她离家生,幸好被消防官兵救下,14日,她又带着孩子离开了大朗,不知去向。“我还年轻,能打工养活她。”杨丽的父亲告诉羊城晚报记者,他希望女儿能想通,回到父母身边。不愿工厂上班时,她认识了 现在的老公,两人很快走到了一起,并有了孩子。杨父原本不同意两人在一起,但女儿肚子大了,只能点头。就这样,17岁的杨丽有了自己的女儿,孩在家中照顾小孩打理家务,而丈夫则在惠州从事手机销售的工作,杨丽就跟父母住在一起。杨父称,在女儿照顾小孩的两年多时间里,女儿从未跟他们提过要出去打工。但出去走走 看看,见见“外面的世界”。杨父便辞去了自己的工作,回来照顾外孙女,女儿则进入大朗一家工厂上班。做到10月底,女儿便辞工不干了。老杨不知道女儿辞工的他猜测是吃不起苦的原因。辞工后,女儿离家出走了一段时间,直到11月12日才回来,询问后才知道女儿与女婿吵架了,但为何吵架,老杨也不知道。被救后携女离去本月1要求,女儿还为此跟他大吵一架。没想到女儿会为此跳楼。“有人要跳楼了。”一名群众大喊道。老杨寻声望去,要跳楼的正是自己的女儿。随后,他和家人苦口婆心劝说女。”据当时第一个破门而入的消防员回忆,当天,他到达现场后,在隔壁房间的窗户对女子进行了劝说,一番劝说无效后,消防员决定带领官兵强行破门而入。当他来到房门。经过努力,终于进入房内将正准备往下跳的杨丽救了下来。14日,记者来到杨丽的住处,敲门许久也无人应门。电话中,老杨告诉记者,杨丽带着女儿离开了大朗,不知去我还年轻,能打工养活她。”老杨说,他希望女儿能想通,早点回到父母身边。')
[2024-03-04 16:47:15,792] [ WARNING] - `max_length` will be deprecated in future releases, use `max_new_tokens` instead.
['" " 90后" " 女主角杨丽: 我是大大大爱，我要去去去救她']
summarizer('内容提要:因为早早结婚,今年20岁的杨丽(化名)已经是一个三岁小孩的妈妈。今年10月31日,和老公大吵一架的她离家出走,直到11月12日才回到租住转。今年10月31日,和老公大吵一架的她离家生,幸好被消防官兵救下,14日,她又带着孩子离开了大朗,不知去向。我还年轻,能打工养活她。杨丽的父亲告诉羊城晚报记工厂上班时,她认识了 现在的老公,两人很快走到了一起,并有了孩子。杨父原本不同意两人在一起,但女儿肚子大了,只能点头。就这样,17岁的杨丽有了自己的女儿,孩跟他们提过要出去打工。但出去走走 看看,见见外面的世界。杨父便辞去了自己的工作,回来照顾外孙女,女儿则进入大朗一家工厂上班。做到10月底,女儿便辞工不干了他猜测是吃不起苦的原因。辞工后,女儿离家出走了一段时间,直到11月12日才回来,询问后才知道女儿与女婿吵架了,但为何吵架,老杨也不知道。被救后携女离去本月1口婆心劝说女 。据当时第一个破门而入的消防员回忆,当天,他到达现场后,在隔壁房间的窗户对女子进行了劝说,一番劝说无效后,消防员决定带领官兵强行破门而入。。经过努力,终于进入房内将正准备往下跳的杨丽救了下来。14日,记者来到杨丽的住处,敲门许久也无人应门。电话中,老杨告诉记者,杨丽带着女儿离开了大朗,不知去边。')
[2024-03-04 16:50:37,158] [ WARNING] - `max_length` will be deprecated in future releases, use `max_new_tokens` instead.
['" " 大大大小小小" " 的“小小小”新妈妈']
'''