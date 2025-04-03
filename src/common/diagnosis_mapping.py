DIAGNOSIS_TO_NAME = {
    'NV': 'Melanocytic nevus',
    'MEL': 'Melanoma',
    'BCC': 'Basal cell carcinoma',
    'BKL': 'Benign keratosis',
    'AK': 'Actinic keratosis',
    'SCC': 'Squamous cell carcinoma',
    'VASC': 'Vascular lesion',
    'DF': 'Dermatofibroma',
    'UNK': 'None of the others',
}

NAME_TO_DIAGNOSIS = {
    'Melanocytic nevus': 'NV',
    'Melanoma': 'MEL',
    'Basal cell carcinoma': 'BCC',
    'Benign keratosis': 'BKL',
    'Actinic keratosis': 'AK',
    'Squamous cell carcinoma': 'SCC',
    'Vascular lesion': 'VASC',
    'Dermatofibroma': 'DF',
    'None of the others': 'UNK',
}

TARGET_TO_DIAGNOSIS = {
    0: 'NV',
    1: 'MEL',
    2: 'BCC',
    3: 'BKL',
    4: 'AK',
    5: 'SCC',
    6: 'VASC',
    7: 'DF',
    8: 'UNK',
}

DIAGNOSIS_TO_TARGET = {
    'NV': 0,
    'MEL': 1,
    'BCC': 2,
    'BKL': 3,
    'AK': 4,
    'SCC': 5,
    'VASC': 6,
    'DF': 7,
    'UNK': 8,
}

MALIGNANT_DIAGNOSIS = [
    'MEL',
    'BCC',
    'AK',
    'SCC',
]

BENIGN_DIAGNOSIS = [
    'NV',
    'BKL',
    'VASC',
    'DF',
    'UNK',
]
