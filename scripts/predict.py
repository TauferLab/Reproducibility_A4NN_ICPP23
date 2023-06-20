import sys
import os
import numpy as np
# sys.path.insert(0, os.environ['NSGA_NET_PATH'])

import torch
from torch import package
import PD_dataset as pd_data
import time
import argparse
import pandas as pd

if torch.cuda.is_available():
    device='cuda'
else:
    device='cpu'

parser = argparse.ArgumentParser("Multi-objective Genetic Algorithm for NAS")
parser.add_argument('--model_path', type=str, help='where is the model file', required=True)
parser.add_argument('--data_path', type=str, help='where is the data root', required=True)
parser.add_argument('--out_path', type=str, default=None, help='where do you want the results of this to go')
args = parser.parse_args()

class Predict(object):
    def __init__(self, path, data_root, outpath):
        self.path = path
        self.data_root = data_root

        if outpath != None:
            self.outpath = outpath
        else: 
            self.outpath = '/'.join(path.split('/')[:-2])

        self.predictions_path = os.path.join(self.outpath, "predictions.csv")
        self.analysis_path = os.path.join(self.outpath, "prediction_analysis.txt")

        rm_start = time.time()
        self.read_model()
        rm_time = time.time() - rm_start
        
        gd_start = time.time()
        self.get_data()
        gd_time = time.time() - gd_start

        run_start = time.time()     
        correct, incorrect, accuracy = self.run_model()
        run_time = time.time() - run_start

        prediction_time_per_image = run_time/len(self.data)

        with open(self.analysis_path, "w") as f:
            f.write(f"Model path is {self.path}.\n")
            f.write(f"Data path is {self.data_root}.\n")
            f.write(f"Read model in {rm_time} seconds.\n")
            f.write(f"Get data in {gd_time} seconds.\n")
            f.write(f"Run model predictions in {run_time} seconds.\n")
            f.write(f"Average prediction time per image is {prediction_time_per_image} seconds.\n")
            f.write(f"Correct: {correct}\tIncorrect: {incorrect}\nAccuracy: {accuracy}\n")

        return

    def get_data(self):
        self.data = pd_data.ProteinDiffDataset(self.data_root, train=False, regression=False)
        return

    def read_model(self):
        importer = package.PackageImporter(self.path)
        print(importer.file_structure())

        mod = importer.import_module('models')
        arch = self.path.split('/')[-1].split('_')[1]
        epoch =  self.path.split('/')[-1].split('_')[3].split('.')[0].strip()

        package_name = f"arch_{arch}_epoch_{epoch}"
        resource_name = f"arch_{arch}_epoch_{epoch}.pkl"
         
        self.loaded_model = importer.load_pickle(package_name, resource_name, map_location=torch.device(device))

        assert package.is_from_package(mod)
        assert package.is_from_package(self.loaded_model)
        return

    def run_model(self):
        self.loaded_model.eval()

        correct, correct_pred, total = 0, 0, 0
        header = ["prediction", "truth"]
        results = []

        with torch.no_grad():
            for step, (inputs, targets) in enumerate(self.data):
                inputs = inputs.unsqueeze(0)
                inputs = inputs.to(device)
                outputs, _ = self.loaded_model(inputs)
                _, predicted = outputs.max(1)
                correct = predicted.eq(targets)
                results.append([predicted.item(), targets])

                if correct.item():
                    correct_pred += 1               
 
        total = len(self.data)
        pd.DataFrame(results).to_csv(self.predictions_path, header=header)
        print(f"Correct: {correct_pred}\tIncorrect: {total - correct_pred}\nAccuracy: {(correct_pred / total) * 100}")
        return correct_pred, total - correct_pred, (correct_pred / total) * 100


if __name__=="__main__":
    start = time.time()   
    Predict(args.model_path, args.data_path, args.out_path)
    total = time.time() - start
    print(f"Total run time: {total} seconds\n")
