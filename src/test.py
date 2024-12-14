from tif_utils import *
from config import *

feature = "sample_0006_4featuresample.tif"
label = "sample_0006_4labelsample.tif"

prob = config.training_prob

prob["p_hflip"] = 0.75;
prob["p_vflip"] = 0;
prob["p_90rot"] = 0;

transform_patch(feature, label, config.training_prob, False,"hv.tif", 
                "./test")

