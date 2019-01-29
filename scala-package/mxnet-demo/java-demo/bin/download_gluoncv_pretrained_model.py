import argparse
import os
from gluoncv.model_zoo import get_model
from gluoncv.utils import export_block

parser = argparse.ArgumentParser(description='Download GluonCV pre-trained model without pre-processing')
parser.add_argument('--model', type=str, required=True,
                    help='name of the model to use')
parser.add_argument('--save-file', type=str, default='',
                    help='path to save the hybrid model')

opt = parser.parse_args()

model_name = opt.model
model_save_path = opt.save_file
file_path = os.path.join(model_save_path, model_name)

net = get_model(model_name, pretrained=True)
export_block(file_path, net, preprocess=False, layout='CHW')
