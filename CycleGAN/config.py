import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DOMAIN_A_DIR = r"C:\Users\pm15334\ra\ParasiticEggDetection\dataset_samsung"
#DOMAIN_A_DIR = r"C:\Users\jazma\RA\dataset_samsung\large_egg\fbs04.jpg"
DOMAIN_B_DIR = r"C:\Users\pm15334\ra\ParasiticEggDetection\dataset_canon"
ANNOTATIONS_PATH = r"C:\Users\pm15334\ra\ParasiticEggDetection\dataset_samsung\Annotations.json"
VAL_DIR = "data/val"
BATCH_SIZE = 1
LEARNING_RATE = 1e-5
LAMBDA_IDENTITY = 0.0
LAMBDA_CYCLE = 10
NUM_WORKERS = 4
NUM_EPOCHS = 150
LOAD_MODEL = True
SAVE_MODEL = True
MODEL_PATH = r'C:\Users\pm15334\ra\ParasiticEggDetection\blur2clean'
CHECKPOINT_GEN_A = "genA.pth.tar"
CHECKPOINT_GEN_B = "genB.pth.tar"
CHECKPOINT_DISC_A = "discA.pth.tar"
CHECKPOINT_DISC_B = "discB.pth.tar"
OUTPUT_PATH = r'C:\Users\pm15334\ra\ParasiticEggDetection\superresolution\newmicroscope'
REPLICATE_DIRECTORY = True

root_path = ["C:\\Users\\pm15334\\ra\\ParasiticEggDetection\\datasets\\01-11-2021_canon",
           "C:\\Users\\pm15334\\ra\\ParasiticEggDetection\\datasets\\01-11-2021_ip13",
           "C:\\Users\\pm15334\\ra\\ParasiticEggDetection\\datasets\\01-11-2021_nikon_dsi2",
           "C:\\Users\\pm15334\\ra\\ParasiticEggDetection\\datasets\\12-7-2021_c",
           "C:\\Users\\pm15334\\ra\\ParasiticEggDetection\\datasets\\12-7-2021_s",
           "C:\\Users\\pm15334\\ra\\ParasiticEggDetection\\datasets\\25-9-2021_nikon",
           "C:\\Users\\pm15334\\ra\\ParasiticEggDetection\\datasets\\25-9-2021_sumsu"]

root_path = [r'C:\Users\pm15334\ra\ParasiticEggDetection\datasets\01-11-2021_canon']
root_path = r'C:\Users\pm15334\ra\ParasiticEggDetection\datasets\newpicturemicroscope'
colour = True