from pathlib import Path

# Root directory
ROOT_DIR = Path(__file__).parent.parent.parent

# Subdirectories
BUFFER = ROOT_DIR / 'buffer'
DATA = ROOT_DIR / 'data'
WORK_DIR = ROOT_DIR / 'work_dir'
RESULTS_DIR = ROOT_DIR / 'results'
MODEL_PATH = ROOT_DIR / 'model'
MODEL_PATH_MULTI = MODEL_PATH  / 'multi'
MODEL_PATH_BINARY = MODEL_PATH / 'binary'

# ISIC data
ISIC_META_DATA = f'{DATA}/ISIC_Complete_No_Duplicates_Binary.csv'
ISIC_DATASET_MAPPING = {
    '2020_train': f'{DATA}/2020/ISIC_2020_Training_JPEG',
    '2019_train': f'{DATA}/2019/ISIC_2019_Training_Input',
    '2018_train': f'{DATA}/2018/ISIC2018_Task3_Training_Input',
    '2018_test': f'{DATA}/2018/ISIC2018_Task3_Test_Input',
    '2018_val': f'{DATA}/2018/ISIC2018_Task3_Validation_Input',
}

SALIENCY_MAP_PATH = f'{BUFFER}/saliency_map.png'
IMG_PRED_INPUT_PATH = f'{BUFFER}/input_img_pred.jpg'
IMG_REL_INPUT_PATH = f'{BUFFER}/input_img_rel.jpg'
IMG_CE_B_REL_PATH = f'{BUFFER}/ce_b_rel.jpg'
IMG_CE_M_REL = f'{BUFFER}/ce_m_rel.jpg'

EVAL_CSV_PATH = f'{BUFFER}/eval.csv'
