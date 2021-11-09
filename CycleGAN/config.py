import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DOMAIN_A_DIR = r"C:\Users\pm15334\ra\ParasiticEggDetection\dataset_samsung"
#DOMAIN_A_DIR = r"C:\Users\pm15334\ra\ParasiticEggDetection\dataset_samsung\large_egg\fbs04.jpg"
DOMAIN_B_DIR = r"C:\Users\pm15334\ra\ParasiticEggDetection\dataset_samsung"
ANNOTATIONS_PATH = r"C:\Users\pm15334\ra\ParasiticEggDetection\dataset_samsung\Annotations_6classes.json"
VAL_DIR = "data/val"
BATCH_SIZE = 1
LEARNING_RATE = 1e-5
LAMBDA_IDENTITY = 0.0
LAMBDA_CYCLE = 10
NUM_WORKERS = 4
NUM_EPOCHS = 50
LOAD_MODEL = True
SAVE_MODEL = True
CHECKPOINT_GEN_A = "genA.pth.tar"
CHECKPOINT_GEN_B = "genB.pth.tar"
CHECKPOINT_DISC_A = "discA.pth.tar"
CHECKPOINT_DISC_B = "discB.pth.tar"
OUTPUT_PATH = r'C:\Users\pm15334\ra\ParasiticEggDetection\samsung2canon'
transforms = A.Compose(
    [
        A.Resize(width=256, height=256),
        A.HorizontalFlip(p=0.5),
        A.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], max_pixel_value=255),
        ToTensorV2(),
     ],
    additional_targets={"image0": "image"},
)