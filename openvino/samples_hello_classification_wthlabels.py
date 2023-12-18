"""
omz_downloader --name alexnet
#omz_converter --name alexnet --output_dir=$PWD
#fz没用，设置域名ip以后可以下载
#omz_converter没法控制输出目录，直接用原始python模块：
python -- /Volumes/data/envs/openvino/bin/mo \
  --framework=caffe \
  --output_dir=$PWD \
  --model_name=alexnet \
  --input=data '--mean_values=data[104.0,117.0,123.0]' \
  --output=prob \
  --input_model=/Volumes/data/workspace/shouxieai/openvino/public/alexnet/alexnet.caffemodel \
  --input_proto=/Volumes/data/workspace/shouxieai/openvino/public/alexnet/alexnet.prototxt \
  '--layout=data(NCHW)' \
  '--input_shape=[1, 3, 227, 227]' \
  --compress_to_fp16=True
ls alexnet* -l
"""
import logging
import logging as log
import sys

import cv2
import numpy as np
from openvino.preprocess import PrePostProcessor, ResizeAlgorithm
from openvino.runtime import Core, Layout, Type

import time

"""
alexnet_labels = ['tench', 'goldfish', 'great white shark', 'tiger shark', 'hammerhead', 'electric ray', 'stingray', 'cock', 'hen', 'ostrich', 'brambling', 'goldfinch', 'house finch', 'junco', 'indigo bunting', 'robin', 'bulbul', 'jay', 'magpie', 'chickadee', 'water ouzel', 'kite', 'bald eagle', 'vulture', 'great grey owl', 'European fire salamander', 'common newt', 'eft', 'spotted salamander', 'axolotl', 'bullfrog', 'tree frog', 'tailed frog', 'loggerhead', 'leatherback turtle', 'mud turtle', 'terrapin', 'box turtle', 'banded gecko', 'common iguana', 'American chameleon', 'whiptail', 'agama', 'frilled lizard', 'alligator lizard', 'Gila monster', 'green lizard', 'African chameleon', 'Komodo dragon', 'African crocodile', 'American alligator', 'triceratops', 'thunder snake', 'ringneck snake', 'hognose snake', 'green snake', 'king snake', 'garter snake', 'water snake', 'vine snake', 'night snake', 'boa constrictor', 'rock python', 'Indian cobra', 'green mamba', 'sea snake', 'horned viper', 'diamondback', 'sidewinder', 'trilobite', 'harvestman', 'scorpion', 'black and gold garden spider', 'barn spider', 'garden spider', 'black widow', 'tarantula', 'wolf spider', 'tick', 'centipede', 'black grouse', 'ptarmigan', 'ruffed grouse', 'prairie chicken', 'peacock', 'quail', 'partridge', 'African grey', 'macaw', 'sulphur-crested cockatoo', 'lorikeet', 'coucal', 'bee eater', 'hornbill', 'hummingbird', 'jacamar', 'toucan', 'drake', 'red-breasted merganser', 'goose', 'black swan', 'tusker', 'echidna', 'platypus', 'wallaby', 'koala', 'wombat', 'jellyfish', 'sea anemone', 'brain coral', 'flatworm', 'nematode', 'conch', 'snail', 'slug', 'sea slug', 'chiton', 'chambered nautilus', 'Dungeness crab', 'rock crab', 'fiddler crab', 'king crab', 'American lobster', 'spiny lobster', 'crayfish', 'hermit crab', 'isopod', 'white stork', 'black stork', 'spoonbill', 'flamingo', 'little blue heron', 'American egret', 'bittern', 'crane bird', 'limpkin', 'European gallinule', 'American coot', 'bustard', 'ruddy turnstone', 'red-backed sandpiper', 'redshank', 'dowitcher', 'oystercatcher', 'pelican', 'king penguin', 'albatross', 'grey whale', 'killer whale', 'dugong', 'sea lion', 'Chihuahua', 'Japanese spaniel', 'Maltese dog', 'Pekinese', 'Shih-Tzu', 'Blenheim spaniel', 'papillon', 'toy terrier', 'Rhodesian ridgeback', 'Afghan hound', 'basset', 'beagle', 'bloodhound', 'bluetick', 'black-and-tan coonhound', 'Walker hound', 'English foxhound', 'redbone', 'borzoi', 'Irish wolfhound', 'Italian greyhound', 'whippet', 'Ibizan hound', 'Norwegian elkhound', 'otterhound', 'Saluki', 'Scottish deerhound', 'Weimaraner', 'Staffordshire bullterrier', 'American Staffordshire terrier', 'Bedlington terrier', 'Border terrier', 'Kerry blue terrier', 'Irish terrier', 'Norfolk terrier', 'Norwich terrier', 'Yorkshire terrier', 'wire-haired fox terrier', 'Lakeland terrier', 'Sealyham terrier', 'Airedale', 'cairn', 'Australian terrier', 'Dandie Dinmont', 'Boston bull', 'miniature schnauzer', 'giant schnauzer', 'standard schnauzer', 'Scotch terrier', 'Tibetan terrier', 'silky terrier', 'soft-coated wheaten terrier', 'West Highland white terrier', 'Lhasa', 'flat-coated retriever', 'curly-coated retriever', 'golden retriever', 'Labrador retriever', 'Chesapeake Bay retriever', 'German short-haired pointer', 'vizsla', 'English setter', 'Irish setter', 'Gordon setter', 'Brittany spaniel', 'clumber', 'English springer', 'Welsh springer spaniel', 'cocker spaniel', 'Sussex spaniel', 'Irish water spaniel', 'kuvasz', 'schipperke', 'groenendael', 'malinois', 'briard', 'kelpie', 'komondor', 'Old English sheepdog', 'Shetland sheepdog', 'collie', 'Border collie', 'Bouvier des Flandres', 'Rottweiler', 'German shepherd', 'Doberman', 'miniature pinscher', 'Greater Swiss Mountain dog', 'Bernese mountain dog', 'Appenzeller', 'EntleBucher', 'boxer', 'bull mastiff', 'Tibetan mastiff', 'French bulldog', 'Great Dane', 'Saint Bernard', 'Eskimo dog', 'malamute', 'Siberian husky', 'dalmatian', 'affenpinscher', 'basenji', 'pug', 'Leonberg', 'Newfoundland', 'Great Pyrenees', 'Samoyed', 'Pomeranian', 'chow', 'keeshond', 'Brabancon griffon', 'Pembroke', 'Cardigan', 'toy poodle', 'miniature poodle', 'standard poodle', 'Mexican hairless', 'timber wolf', 'white wolf', 'red wolf', 'coyote', 'dingo', 'dhole', 'African hunting dog', 'hyena', 'red fox', 'kit fox', 'Arctic fox', 'grey fox', 'tabby', 'tiger cat', 'Persian cat', 'Siamese cat', 'Egyptian cat', 'cougar', 'lynx', 'leopard', 'snow leopard', 'jaguar', 'lion', 'tiger', 'cheetah', 'brown bear', 'American black bear', 'ice bear', 'sloth bear', 'mongoose', 'meerkat', 'tiger beetle', 'ladybug', 'ground beetle', 'long-horned beetle', 'leaf beetle', 'dung beetle', 'rhinoceros beetle', 'weevil', 'fly', 'bee', 'ant', 'grasshopper', 'cricket', 'walking stick', 'cockroach', 'mantis', 'cicada', 'leafhopper', 'lacewing', 'dragonfly', 'damselfly', 'admiral', 'ringlet', 'monarch', 'cabbage butterfly', 'sulphur butterfly', 'lycaenid', 'starfish', 'sea urchin', 'sea cucumber', 'wood rabbit', 'hare', 'Angora', 'hamster', 'porcupine', 'fox squirrel', 'marmot', 'beaver', 'guinea pig', 'sorrel', 'zebra', 'hog', 'wild boar', 'warthog', 'hippopotamus', 'ox', 'water buffalo', 'bison', 'ram', 'bighorn', 'ibex', 'hartebeest', 'impala', 'gazelle', 'Arabian camel', 'llama', 'weasel', 'mink', 'polecat', 'black-footed ferret', 'otter', 'skunk', 'badger', 'armadillo', 'three-toed sloth', 'orangutan', 'gorilla', 'chimpanzee', 'gibbon', 'siamang', 'guenon', 'patas', 'baboon', 'macaque', 'langur', 'colobus', 'proboscis monkey', 'marmoset', 'capuchin', 'howler monkey', 'titi', 'spider monkey', 'squirrel monkey', 'Madagascar cat', 'indri', 'Indian elephant', 'African elephant', 'lesser panda', 'giant panda', 'barracouta', 'eel', 'coho', 'rock beauty', 'anemone fish', 'sturgeon', 'gar', 'lionfish', 'puffer', 'abacus', 'abaya', 'academic gown', 'accordion', 'acoustic guitar', 'aircraft carrier', 'airliner', 'airship', 'altar', 'ambulance', 'amphibian', 'analog clock', 'apiary', 'apron', 'ashcan', 'assault rifle', 'backpack', 'bakery', 'balance beam', 'balloon', 'ballpoint', 'Band Aid', 'banjo', 'bannister', 'barbell', 'barber chair', 'barbershop', 'barn', 'barometer', 'barrel', 'barrow', 'baseball', 'basketball', 'bassinet', 'bassoon', 'bathing cap', 'bath towel', 'bathtub', 'beach wagon', 'beacon', 'beaker', 'bearskin', 'beer bottle', 'beer glass', 'bell cote', 'bib', 'bicycle-built-for-two', 'bikini', 'binder', 'binoculars', 'birdhouse', 'boathouse', 'bobsled', 'bolo tie', 'bonnet', 'bookcase', 'bookshop', 'bottlecap', 'bow', 'bow tie', 'brass', 'brassiere', 'breakwater', 'breastplate', 'broom', 'bucket', 'buckle', 'bulletproof vest', 'bullet train', 'butcher shop', 'cab', 'caldron', 'candle', 'cannon', 'canoe', 'can opener', 'cardigan', 'car mirror', 'carousel', "carpenter's kit", 'carton', 'car wheel', 'cash machine', 'cassette', 'cassette player', 'castle', 'catamaran', 'CD player', 'cello', 'cellular telephone', 'chain', 'chainlink fence', 'chain mail', 'chain saw', 'chest', 'chiffonier', 'chime', 'china cabinet', 'Christmas stocking', 'church', 'cinema', 'cleaver', 'cliff dwelling', 'cloak', 'clog', 'cocktail shaker', 'coffee mug', 'coffeepot', 'coil', 'combination lock', 'computer keyboard', 'confectionery', 'container ship', 'convertible', 'corkscrew', 'cornet', 'cowboy boot', 'cowboy hat', 'cradle', 'crane', 'crash helmet', 'crate', 'crib', 'Crock Pot', 'croquet ball', 'crutch', 'cuirass', 'dam', 'desk', 'desktop computer', 'dial telephone', 'diaper', 'digital clock', 'digital watch', 'dining table', 'dishrag', 'dishwasher', 'disk brake', 'dock', 'dogsled', 'dome', 'doormat', 'drilling platform', 'drum', 'drumstick', 'dumbbell', 'Dutch oven', 'electric fan', 'electric guitar', 'electric locomotive', 'entertainment center', 'envelope', 'espresso maker', 'face powder', 'feather boa', 'file', 'fireboat', 'fire engine', 'fire screen', 'flagpole', 'flute', 'folding chair', 'football helmet', 'forklift', 'fountain', 'fountain pen', 'four-poster', 'freight car', 'French horn', 'frying pan', 'fur coat', 'garbage truck', 'gasmask', 'gas pump', 'goblet', 'go-kart', 'golf ball', 'golfcart', 'gondola', 'gong', 'gown', 'grand piano', 'greenhouse', 'grille', 'grocery store', 'guillotine', 'hair slide', 'hair spray', 'half track', 'hammer', 'hamper', 'hand blower', 'hand-held computer', 'handkerchief', 'hard disc', 'harmonica', 'harp', 'harvester', 'hatchet', 'holster', 'home theater', 'honeycomb', 'hook', 'hoopskirt', 'horizontal bar', 'horse cart', 'hourglass', 'iPod', 'iron', "jack-o'-lantern", 'jean', 'jeep', 'jersey', 'jigsaw puzzle', 'jinrikisha', 'joystick', 'kimono', 'knee pad', 'knot', 'lab coat', 'ladle', 'lampshade', 'laptop', 'lawn mower', 'lens cap', 'letter opener', 'library', 'lifeboat', 'lighter', 'limousine', 'liner', 'lipstick', 'Loafer', 'lotion', 'loudspeaker', 'loupe', 'lumbermill', 'magnetic compass', 'mailbag', 'mailbox', 'maillot', 'maillot tank suit', 'manhole cover', 'maraca', 'marimba', 'mask', 'matchstick', 'maypole', 'maze', 'measuring cup', 'medicine chest', 'megalith', 'microphone', 'microwave', 'military uniform', 'milk can', 'minibus', 'miniskirt', 'minivan', 'missile', 'mitten', 'mixing bowl', 'mobile home', 'Model T', 'modem', 'monastery', 'monitor', 'moped', 'mortar', 'mortarboard', 'mosque', 'mosquito net', 'motor scooter', 'mountain bike', 'mountain tent', 'mouse', 'mousetrap', 'moving van', 'muzzle', 'nail', 'neck brace', 'necklace', 'nipple', 'notebook', 'obelisk', 'oboe', 'ocarina', 'odometer', 'oil filter', 'organ', 'oscilloscope', 'overskirt', 'oxcart', 'oxygen mask', 'packet', 'paddle', 'paddlewheel', 'padlock', 'paintbrush', 'pajama', 'palace', 'panpipe', 'paper towel', 'parachute', 'parallel bars', 'park bench', 'parking meter', 'passenger car', 'patio', 'pay-phone', 'pedestal', 'pencil box', 'pencil sharpener', 'perfume', 'Petri dish', 'photocopier', 'pick', 'pickelhaube', 'picket fence', 'pickup', 'pier', 'piggy bank', 'pill bottle', 'pillow', 'ping-pong ball', 'pinwheel', 'pirate', 'pitcher', 'plane', 'planetarium', 'plastic bag', 'plate rack', 'plow', 'plunger', 'Polaroid camera', 'pole', 'police van', 'poncho', 'pool table', 'pop bottle', 'pot', "potter's wheel", 'power drill', 'prayer rug', 'printer', 'prison', 'projectile', 'projector', 'puck', 'punching bag', 'purse', 'quill', 'quilt', 'racer', 'racket', 'radiator', 'radio', 'radio telescope', 'rain barrel', 'recreational vehicle', 'reel', 'reflex camera', 'refrigerator', 'remote control', 'restaurant', 'revolver', 'rifle', 'rocking chair', 'rotisserie', 'rubber eraser', 'rugby ball', 'rule', 'running shoe', 'safe', 'safety pin', 'saltshaker', 'sandal', 'sarong', 'sax', 'scabbard', 'scale', 'school bus', 'schooner', 'scoreboard', 'screen', 'screw', 'screwdriver', 'seat belt', 'sewing machine', 'shield', 'shoe shop', 'shoji', 'shopping basket', 'shopping cart', 'shovel', 'shower cap', 'shower curtain', 'ski', 'ski mask', 'sleeping bag', 'slide rule', 'sliding door', 'slot', 'snorkel', 'snowmobile', 'snowplow', 'soap dispenser', 'soccer ball', 'sock', 'solar dish', 'sombrero', 'soup bowl', 'space bar', 'space heater', 'space shuttle', 'spatula', 'speedboat', 'spider web', 'spindle', 'sports car', 'spotlight', 'stage', 'steam locomotive', 'steel arch bridge', 'steel drum', 'stethoscope', 'stole', 'stone wall', 'stopwatch', 'stove', 'strainer', 'streetcar', 'stretcher', 'studio couch', 'stupa', 'submarine', 'suit', 'sundial', 'sunglass', 'sunglasses', 'sunscreen', 'suspension bridge', 'swab', 'sweatshirt', 'swimming trunks', 'swing', 'switch', 'syringe', 'table lamp', 'tank', 'tape player', 'teapot', 'teddy', 'television', 'tennis ball', 'thatch', 'theater curtain', 'thimble', 'thresher', 'throne', 'tile roof', 'toaster', 'tobacco shop', 'toilet seat', 'torch', 'totem pole', 'tow truck', 'toyshop', 'tractor', 'trailer truck', 'tray', 'trench coat', 'tricycle', 'trimaran', 'tripod', 'triumphal arch', 'trolleybus', 'trombone', 'tub', 'turnstile', 'typewriter keyboard', 'umbrella', 'unicycle', 'upright', 'vacuum', 'vase', 'vault', 'velvet', 'vending machine', 'vestment', 'viaduct', 'violin', 'volleyball', 'waffle iron', 'wall clock', 'wallet', 'wardrobe', 'warplane', 'washbasin', 'washer', 'water bottle', 'water jug', 'water tower', 'whiskey jug', 'whistle', 'wig', 'window screen', 'window shade', 'Windsor tie', 'wine bottle', 'wing', 'wok', 'wooden spoon', 'wool', 'worm fence', 'wreck', 'yawl', 'yurt', 'web site', 'comic book', 'crossword puzzle', 'street sign', 'traffic light', 'book jacket', 'menu', 'plate', 'guacamole', 'consomme', 'hot pot', 'trifle', 'ice cream', 'ice lolly', 'French loaf', 'bagel', 'pretzel', 'cheeseburger', 'hotdog', 'mashed potato', 'head cabbage', 'broccoli', 'cauliflower', 'zucchini', 'spaghetti squash', 'acorn squash', 'butternut squash', 'cucumber', 'artichoke', 'bell pepper', 'cardoon', 'mushroom', 'Granny Smith', 'strawberry', 'orange', 'lemon', 'fig', 'pineapple', 'banana', 'jackfruit', 'custard apple', 'pomegranate', 'hay', 'carbonara', 'chocolate sauce', 'dough', 'meat loaf', 'pizza', 'potpie', 'burrito', 'red wine', 'espresso', 'cup', 'eggnog', 'alp', 'bubble', 'cliff', 'coral reef', 'geyser', 'lakeside', 'promontory', 'sandbar', 'seashore', 'valley', 'volcano', 'ballplayer', 'groom', 'scuba diver', 'rapeseed', 'daisy', "yellow lady's slipper", 'corn', 'acorn', 'hip', 'buckeye', 'coral fungus', 'agaric', 'gyromitra', 'stinkhorn', 'earthstar', 'hen-of-the-woods', 'bolete', 'ear', 'toilet tissue']
with open('labels-alexnet.txt', 'w') as fp:
    fp.write('\n'.join(alexnet_labels))
    fp.close()
"""

def main():
    log.basicConfig(format='[ %(levelname)s ] %(message)s', level=log.INFO, stream=sys.stdout)

    if len(sys.argv) != 5:
        log.info(f'Usage: {sys.argv[0]} <path_to_model> <path_to_image> <device_name> <label_file_name>')
        return 1

    model_path = sys.argv[1]
    image_path = sys.argv[2]
    device_name = sys.argv[3]
    label_file_name = sys.argv[4]
    with open(label_file_name, 'r') as fp:
        labels = [line.replace('\n', '') for line in fp.readlines()]

    log.info('Creating OpenVINO Runtime Core')
    core = Core()

    log.info(f'Reading the model:{model_path}')
    model = core.read_model(model_path)

    if len(model.inputs) != 1:
        log.error('Sample supports only single input topologies')
        return -1

    if len(model.outputs) != 1:
        log.error('Sample supports only single output topologies')
        return -1

    ppp = PrePostProcessor(model)
    ppp.input().model().set_layout(Layout('NCHW'))
    ppp.output().tensor().set_element_type(Type.f32)

    a1 = time.time()
    image = cv2.imread(image_path)
    input_tensor = np.expand_dims(image, 0)

    _, h, w, _ = input_tensor.shape

    ppp.input().tensor() \
        .set_shape(input_tensor.shape) \
        .set_element_type(Type.u8) \
        .set_layout(Layout('NHWC'))

    ppp.input().preprocess().resize(ResizeAlgorithm.RESIZE_LINEAR)

    model = ppp.build()

    #log.info('Loading the model to the plugin')
    compiled_model = core.compile_model(model, device_name)

    #log.info('Starting inference in synchronous mode')
    a2 = time.time()
    results = compiled_model.infer_new_request({0: input_tensor})
    b = time.time()
    log.info(f'---debug---infer time:{b-a2}')
    log.info(f'---debug---preprocess(recompile every time for dynamic batch) + infer time:{b-a1}')

    predictions = next(iter(results.values()))

    probs = predictions.reshape(-1)

    top_10 = np.argsort(probs)[-10:][::-1]

    header = 'class_id        probability'

    log.info(f'Image path: {image_path}')
    log.info('Top 10 results: ')
    log.info(header)
    log.info('-' * len(header))

    for class_id in top_10:
        number_indent = len('class_id       ') - len(str(labels[class_id])) + 1
        probability_indent = ' ' * number_indent
        class_str = str(labels[class_id])
        log.info(f'{class_str}{probability_indent}{probs[class_id]:.7f}')

    log.info('')

    log.info('This sample is an API example, for any performance measurements please use the dedicated benchmark_app tool\n')
    return 0

if __name__ == '__main__':
    sys.exit(main())


"""
python samples_hello_classification_wthlabels.py alexnet.xml car.jpeg CPU labels-alexnet.txt
[ INFO ] Creating OpenVINO Runtime Core
[ INFO ] Reading the model:alexnet.xml
[ INFO ] ---debug---infer time:0.05411195755004883
[ INFO ] ---debug---preprocess(recompile every time for dynamic batch) + infer time:1.7285492420196533
[ INFO ] Image path: car.jpeg
[ INFO ] Top 10 results: 
[ INFO ] class_id        probability
[ INFO ] ---------------------------
[ INFO ] bobsled         0.2675963
[ INFO ] sports car      0.1104873
[ INFO ] airship         0.0519568
[ INFO ] ocarina         0.0464151
[ INFO ] projectile      0.0321938
[ INFO ] waffle iron     0.0316675
[ INFO ] toaster         0.0302557
[ INFO ] whistle         0.0292995
[ INFO ] harmonica       0.0292969
[ INFO ] convertible     0.0278494
[ INFO ] 
[ INFO ] This sample is an API example, for any performance measurements please use the dedicated benchmark_app tool

python samples_hello_classification_wthlabels.py resnet_fr_torch_onnx-static.xml car.jpeg CPU labels-imagenet.txt
[ INFO ] Creating OpenVINO Runtime Core
[ INFO ] Reading the model:resnet_fr_torch_onnx-static.xml
[ INFO ] ---debug---infer time:0.09466099739074707
[ INFO ] ---debug---preprocess(recompile every time for dynamic batch) + infer time:2.5686419010162354
[ INFO ] Image path: car.jpeg
[ INFO ] Top 10 results: 
[ INFO ] class_id        probability
[ INFO ] ---------------------------
[ INFO ] SPORTS CAR      9.1641951
[ INFO ] CANOE           8.6908731
[ INFO ] SCREWDRIVER     8.6520672
[ INFO ] PADDLE          7.4257112
[ INFO ] HAND-HELD COMPUTER7.2755351
[ INFO ] SAFETY PIN      6.8635659
[ INFO ] TRAILER TRUCK   6.8381248
[ INFO ] CONVERTIBLE     6.8183765
[ INFO ] CAN OPENER      6.7357759
[ INFO ] LAWN MOWER      6.6663685
[ INFO ] 
[ INFO ] This sample is an API example, for any performance measurements please use the dedicated benchmark_app tool

python samples_hello_classification_wthlabels.py resnet_fr_torch_onnx-static.xml laptop.jpeg CPU labels-imagenet.txt
[ INFO ] Creating OpenVINO Runtime Core
[ INFO ] Reading the model:resnet_fr_torch_onnx-static.xml
[ INFO ] ---debug---infer time:0.08134102821350098
[ INFO ] ---debug---preprocess(recompile every time for dynamic batch) + infer time:0.3589189052581787
[ INFO ] Image path: laptop.jpeg
[ INFO ] Top 10 results: 
[ INFO ] class_id        probability
[ INFO ] ---------------------------
[ INFO ] NOTEBOOK        12.9085922
[ INFO ] LAPTOP          12.2043066
[ INFO ] DESKTOP COMPUTER11.4831591
[ INFO ] SCREEN          11.0500584
[ INFO ] SPACE BAR       10.8050861
[ INFO ] HAND-HELD COMPUTER10.1209288
[ INFO ] MOUSE           8.8752174
[ INFO ] MONITOR         8.7504263
[ INFO ] DESK            8.2985363
[ INFO ] WEB SITE        8.1583757
[ INFO ] 
[ INFO ] This sample is an API example, for any performance measurements please use the dedicated benchmark_app tool

python samples_hello_classification_wthlabels.py resnet_fr_torch_onnx-static.xml dog.jpeg CPU labels-imagenet.txt
[ INFO ] Creating OpenVINO Runtime Core
[ INFO ] Reading the model:resnet_fr_torch_onnx-static.xml
[ INFO ] ---debug---infer time:0.0819697380065918
[ INFO ] ---debug---preprocess(recompile every time for dynamic batch) + infer time:0.36423778533935547
[ INFO ] Image path: dog.jpeg
[ INFO ] Top 10 results: 
[ INFO ] class_id        probability
[ INFO ] ---------------------------
[ INFO ] PEMBROKE        16.7581329
[ INFO ] CARDIGAN        15.4404440
[ INFO ] DINGO           8.4343081
[ INFO ] CHIHUAHUA       8.1803379
[ INFO ] BASENJI         8.1641273
[ INFO ] COLLIE          8.1298742
[ INFO ] PAPILLON        8.0236664
[ INFO ] KELPIE          7.6285148
[ INFO ] BORDER COLLIE   7.5138373
[ INFO ] NORWICH TERRIER 7.4527345
[ INFO ] 
[ INFO ] This sample is an API example, for any performance measurements please use the dedicated benchmark_app tool

python samples_hello_classification_wthlabels.py resnet_fr_torch_onnx-fp16-static.xml car.jpeg CPU labels-imagenet.txt
[ INFO ] Creating OpenVINO Runtime Core
[ INFO ] Reading the model:resnet_fr_torch_onnx-fp16-static.xml
[ INFO ] ---debug---infer time:0.08330583572387695
[ INFO ] ---debug---preprocess(recompile every time for dynamic batch) + infer time:0.4004368782043457
[ INFO ] Image path: car.jpeg
[ INFO ] Top 10 results: 
[ INFO ] class_id        probability
[ INFO ] ---------------------------
[ INFO ] SPORTS CAR      9.1572018
[ INFO ] CANOE           8.6955929
[ INFO ] SCREWDRIVER     8.6612892
[ INFO ] PADDLE          7.4297504
[ INFO ] HAND-HELD COMPUTER7.2736182
[ INFO ] SAFETY PIN      6.8695626
[ INFO ] TRAILER TRUCK   6.8370962
[ INFO ] CONVERTIBLE     6.8107276
[ INFO ] CAN OPENER      6.7365770
[ INFO ] LAWN MOWER      6.6672306
[ INFO ] 
[ INFO ] This sample is an API example, for any performance measurements please use the dedicated benchmark_app tool

python samples_hello_classification_wthlabels.py resnet_fr_torch_onnx-static.xml car.jpeg CPU labels-imagenet.txt
[ INFO ] Creating OpenVINO Runtime Core
[ INFO ] Reading the model:resnet_fr_torch_onnx-static.xml
[ INFO ] ---debug---infer time:0.08616805076599121
[ INFO ] ---debug---preprocess(recompile every time for dynamic batch) + infer time:0.33767104148864746
[ INFO ] Image path: car.jpeg
[ INFO ] Top 10 results: 
[ INFO ] class_id        probability
[ INFO ] ---------------------------
[ INFO ] SPORTS CAR      9.1641951
[ INFO ] CANOE           8.6908731
[ INFO ] SCREWDRIVER     8.6520672
[ INFO ] PADDLE          7.4257112
[ INFO ] HAND-HELD COMPUTER7.2755351
[ INFO ] SAFETY PIN      6.8635659
[ INFO ] TRAILER TRUCK   6.8381248
[ INFO ] CONVERTIBLE     6.8183765
[ INFO ] CAN OPENER      6.7357759
[ INFO ] LAWN MOWER      6.6663685
[ INFO ] 
[ INFO ] This sample is an API example, for any performance measurements please use the dedicated benchmark_app tool

"""