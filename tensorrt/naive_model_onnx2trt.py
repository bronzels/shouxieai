import onnx
import tensorrt as trt
import torch

onnx_model = 'naive_model.onnx'

model = onnx.load(onnx_model)

logger = trt.Logger(trt.Logger.ERROR)
builder = trt.Builder(logger)
EXPLICIT_BATCH = 1 << (int)(
    trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
network = builder.create_network(EXPLICIT_BATCH)

parser = trt.OnnxParser(network, logger)

if not parser.parse(model.SerializeToString()):
    error_msg = ''
    for error in range(parser.num_errors):
        error_msg += f'{parser.get_error(error)}\n'
    raise RuntimeError(f'Failed to parse onnx, {error_msg}')

config = builder.create_builder_config()
config.max_workspace_size = 1<<20
profile = builder.create_optimization_profile()

profile.set_shape('input', [1, 3, 224, 224], [1, 3, 224, 224], [1, 3, 224, 224])
config.add_optimization_profile(profile)

device = torch.device('cuda:0')
with torch.cuda.device(device):
    engine = builder.build_engine(network, config)

with open('naive_model.engine', mode='wb') as f:
    f.write(bytearray(engine.serialize()))
    print("generating file done!")

