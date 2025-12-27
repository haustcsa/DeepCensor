
import yaml
from easydict import EasyDict

with open("E:/scpcode/WaveGuard-master14/config/train.yaml", "r") as f:
    training_config = EasyDict(yaml.load(f, Loader=yaml.SafeLoader))
