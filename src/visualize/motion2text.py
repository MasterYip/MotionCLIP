import os
import sys
sys.path.append('.')

import matplotlib.pyplot as plt
import torch
import csv
from src.utils.get_model_and_data import get_model_and_data
from src.parser.visualize import parser
from src.visualize.visualize import viz_motion2text, get_gpu_device
from src.utils.misc import load_model_wo_clip

import src.utils.fixseed  # noqa

plt.switch_backend('agg')


def main():
    # parse options
    parameters, folder, checkpointname, epoch = parser()
    gpu_device = get_gpu_device()
    parameters["device"] = f"cuda:{gpu_device}"
    model, datasets = get_model_and_data(parameters, split='all')

    print("Restore weights..")
    checkpointpath = os.path.join(folder, checkpointname)
    state_dict = torch.load(checkpointpath, map_location=parameters["device"])
    load_model_wo_clip(model, state_dict)

    assert os.path.isfile(parameters['input_file'])
    with open(parameters['input_file'], 'r') as fr:
        motion_csv = list(csv.DictReader(fr))

    text_vocabs = [line['motion_text'] for line in motion_csv]
    viz_motion2text(model, datasets, motion_csv, epoch, parameters, folder=folder, text_vocabulary=text_vocabs)


if __name__ == '__main__':
    main()
