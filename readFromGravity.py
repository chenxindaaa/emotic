from tensorbay import GAS
from tensorbay.dataset import Dataset
import numpy as np
from PIL import Image
import cv2
from tqdm import tqdm
import os

def cat_to_one_hot(y_cat):
    '''
    One hot encode a categorical label.
    :param y_cat: Categorical label.
    :return: One hot encoded categorical label.
    '''
    cat2ind = {'Affection': 0, 'Anger': 1, 'Annoyance': 2, 'Anticipation': 3, 'Aversion': 4,
               'Confidence': 5, 'Disapproval': 6, 'Disconnection': 7, 'Disquietment': 8,
               'Doubt/Confusion': 9, 'Embarrassment': 10, 'Engagement': 11, 'Esteem': 12,
               'Excitement': 13, 'Fatigue': 14, 'Fear': 15, 'Happiness': 16, 'Pain': 17,
               'Peace': 18, 'Pleasure': 19, 'Sadness': 20, 'Sensitivity': 21, 'Suffering': 22,
               'Surprise': 23, 'Sympathy': 24, 'Yearning': 25}
    one_hot_cat = np.zeros(26)
    for em in y_cat:
        one_hot_cat[cat2ind[em]] = 1
    return one_hot_cat

gas = GAS('Accesskey-dac95e0d8e685ef4d5f8b80d51e38499')
dataset = Dataset("Emotic", gas)
segments = dataset.keys()
save_dir = './data/emotic_pre'
if not os.path.exists(save_dir):
    os.makedirs(save_dir)
for seg in ['test', 'val', 'train']:
    segment = dataset[seg]
    context_arr, body_arr, cat_arr, cont_arr = [], [], [], []
    for data in tqdm(segment[:100]):
        with data.open() as fp:
            context = np.asarray(Image.open(fp))
        if len(context.shape) == 2:
            context = cv2.cvtColor(context, cv2.COLOR_GRAY2RGB)
        context_cv = cv2.resize(context, (224, 224))
        for label_box2d in data.label.box2d:
            xmin = label_box2d.xmin
            ymin = label_box2d.ymin
            xmax = label_box2d.xmax
            ymax = label_box2d.ymax
            body = context[ymin:ymax, xmin:xmax]
            body_cv = cv2.resize(body, (128, 128))
            context_arr.append(context_cv)
            body_arr.append(body_cv)
            cont_arr.append(np.array([int(label_box2d.attributes['valence']), int(label_box2d.attributes['arousal']), int(label_box2d.attributes['dominance'])]))
            cat_arr.append(np.array(cat_to_one_hot(label_box2d.attributes['categories'])))
    context_arr = np.array(context_arr)
    body_arr = np.array(body_arr)
    cat_arr = np.array(cat_arr)
    cont_arr = np.array(cont_arr)
    np.save(os.path.join(save_dir, '%s_context_arr.npy' % (seg)), context_arr)
    np.save(os.path.join(save_dir, '%s_body_arr.npy' % (seg)), body_arr)
    np.save(os.path.join(save_dir, '%s_cat_arr.npy' % (seg)), cat_arr)
    np.save(os.path.join(save_dir, '%s_cont_arr.npy' % (seg)), cont_arr)

