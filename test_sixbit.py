from LoadData import LoadData
import models.TfLiteModels as tlmodels
from attacks.AdaptiveSimBA import AdaptiveSimBA
from attacks.SparseSimBA import SparseSimBA
from AttackEngine import AttackEngine
from Logger import Logger
import Utils as utils
import matplotlib.pyplot as plt
from os.path import abspath
import logging
logging.basicConfig(level=logging.INFO)

parentdir = abspath('')
loadData = LoadData(ground_truth_file_path=parentdir + "/data/val.txt", img_folder_path=parentdir + "/data/ILSVRC2012_img_val/")

model = tlmodels.EfficientNetB7(loadData=loadData, bit_width=16)
data_generator = loadData.get_data([(100, 120)], model.metadata)
loadData.save(data_generator, "ILSVRC2012_val_100_to_120_EfficientNetB7")
data_generator = loadData.load("ILSVRC2012_val_100_to_120_EfficientNetB7", [(7, 8)])

attacks = [
  SparseSimBA(size=8, epsilon=200/255)
]

#%%

metrics = [
  'absolute-value norm',
  'psnr',
  'ssim'
]

#%%

attack_engine = AttackEngine(data_generator, model, attacks)
loggers, success_rates = attack_engine.run(metric_names=metrics)
attack_engine.close() # save experiment logs

success_rates

logger = Logger()
logger.load(1)


logs = logger.get_all()


plt.figure(figsize=(10,7))
with plt.style.context('seaborn-whitegrid'):
  plt.plot(logs[0]['ssim'])
