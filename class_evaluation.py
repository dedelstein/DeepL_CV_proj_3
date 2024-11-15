from sklearn.metrics import confusion_matrix
import numpy as np
import torch

def class_eval(model, dataloader, device, unpack_batch):
    # Call the command as "class_eval(model, val_loader, device, unpack_and_process_batch)" for the given model
    mini_batch = 64
    model.eval()
    predictions = []
    truth = []

    with torch.no_grad():
        for batch in dataloader:
            rois, labels = unpack_batch(batch)

            if rois.size(0) == 0:
                continue

            rois = torch.split(rois, mini_batch)
            labels = torch.split(labels, mini_batch)

            for batch_rois, batch_labels in zip(rois, labels):
                batch_rois = batch_rois.to(device)
                batch_labels = batch_labels.to(device)

                outputs = model(batch_rois)
                predicted = outputs.argmax(dim = 1)

                predictions.extend(predicted.cpu().numpy())
                truth.extend(batch_labels.cpu().numpy())

        predictions = np.array(predictions)
        truth = np.array(truth)

    TN, FP, FN, TP = confusion_matrix(truth, predictions).ravel()

    accuracy = (TP + TN) / (TP + TN + FP + FN)
    precision = TP / (TP + FP)
    recall = TP / (TP + FN)
    f1_score = 2 * (precision * recall) / (precision + recall)

    print(f"Accuracy: {accuracy}\tPrecision: {precision}\tRecall: {recall}\tF1-score: {f1_score}")
