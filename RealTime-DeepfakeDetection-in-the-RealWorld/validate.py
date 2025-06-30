import torch
import numpy as np
from networks.LaDeDa import LaDeDa9
from networks.Tiny_LaDeDa import tiny_ladeda
from sklearn.metrics import average_precision_score, precision_recall_curve, accuracy_score, roc_auc_score, precision_score, recall_score
from data import create_dataloader
import torch.nn as nn
#hinzugefügt
import csv
from options.test_options import TestOptions
import os

#def validate(model, data_loader, opt):
def validate(model, opt):
    import time
    data_loader, _ = create_dataloader(opt)
    with torch.no_grad():
        y_true, y_pred = [], []
        for img, label in data_loader:
            img_input = img.cuda()
            y_pred.extend(model(img_input).sigmoid().flatten().tolist())
            y_true.extend(label.flatten().tolist())

    y_true, y_pred = np.array(y_true), np.array(y_pred)
    r_acc = accuracy_score(y_true[y_true == 0], y_pred[y_true == 0] > 0.5)
    f_acc = accuracy_score(y_true[y_true == 1], y_pred[y_true == 1] > 0.5)
    acc = accuracy_score(y_true, y_pred > 0.5)
    ap = average_precision_score(y_true, y_pred)
    # hinzugefügt
    # auc = roc_auc_score(y_true, y_pred)
    try:
        auc = roc_auc_score(y_true, y_pred)
    except ValueError as e:
        if "Only one class present in y_true. ROC AUC score is not defined in that case." in str(e):
            auc = None
        else:
            raise e
    precision = precision_score(y_true, y_pred > 0.5)
    recall = recall_score(y_true, y_pred > 0.5)
    #hinzugefügt
    save_results(acc, ap, r_acc, f_acc, auc, precision, recall)
    return acc, ap, r_acc, f_acc, auc, precision, recall

def save_results(acc, ap, r_acc, f_acc, auc, precision, recall):
    # Mapping von Rohnamen auf lesbare Namen
    name_map = {
        "Midjourney": "MJ",
        "stable_diffusion_v_1_4": "SD1.4",
        "stable_diffusion_v_1_5": "SD1.5",
        "wukong": "Wukong",
        "glide": "GLIDE",
        "ADM": "ADM",
        "VQDM": "VQDM",
        "BigGAN": "BigGAN"
    }

    opt = TestOptions().parse(print_options=False)
    csv_filename = opt.result_path or "default_results.csv"
    train_set = os.path.basename(os.path.dirname(os.path.dirname(opt.checkpoints_dir)))
    test_set = opt.generator
    train_name = name_map.get(train_set, train_set)
    test_name = name_map.get(test_set, test_set)

    with open(csv_filename, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow([
            "Training_Subset", "Test_Subset", "Model", "Accuracy", "Average Precision",
            "Real Image Accuracy", "Fake Image Accuracy", "AUC", "Precision", "Recall"
        ])
        writer.writerow([
            train_name, test_name, "LaDeDa", acc, ap, r_acc, f_acc, auc, precision, recall
        ])

    print(f"Ergebnisse wurden in {csv_filename} gespeichert.")

if __name__ == '__main__':
    opt = TestOptions().parse(print_options=False)
    model = LaDeDa(num_classes=1)
    state_dict = torch.load(opt.model_path, map_location='cpu')
    model.load_state_dict(state_dict['model'])
    model.cuda()
    model.eval()

    acc, ap, r_acc, f_acc, auc, precision, recall = validate(model, opt)
    print("accuracy:", acc)
    print("average precision:", avg_precision)
    print("accuracy of real images:", r_acc)
    print("accuracy of fake images:", f_acc)


