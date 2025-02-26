import os
import numpy as np
from barbar import Bar
import pandas as pd
from tqdm import tqdm
from glob import glob
import imageio.v3 as imageio

import seaborn as sns
import matplotlib.pyplot as plt

import torch
import torchvision.transforms as transforms

#
# FOR TRAINING
#


def run_training(
        train_loader, val_loader, net, num_epochs, criterion, optimizer, scheduler=None, early_stopping=None, device="cpu"
):
    """This function runs the training scheme for the training dataset, and validates on the validation dataset as well

    Arguments:
        train_loader: The training dataloader
        val_loader: The validation dataloader
        net: The model architecture to perform the training on
        num_epochs: The total number of epochs to train the models for
        criterion: The loss function
        optimizer: The optimizer object
        scheduler: The learning rate scheduler (the workflow is suited right now for `ReduceLROnPlateau`)
        early_stopping: The early stopping functionality
        device: The torch device

    Returns:
        accuracy: The accuracy on training set
        val_accuracy: The accuracy on validation set
        losses: The losses on training set (per epoch)
        val_losses: The losses on validation set (per epoch)
    """
    accuracy, val_accuracy = [], []
    losses, val_losses = [], []

    for epoch in range(num_epochs):
        running_loss = 0.0
        correct_total = 0.0
        num_samples_total = 0.0
        for i, data in enumerate(Bar(train_loader)):
            # getting the training inputs
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)

            # setting the parameter gradients to zero
            optimizer.zero_grad()

            # forward pass, backward pass, optimize
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # computing accuracy
            _, predicted = torch.max(outputs, 1)
            b_len, corr = get_accuracy(predicted, labels)
            num_samples_total += b_len
            correct_total += corr
            running_loss += loss.item()

        running_loss /= len(train_loader)
        train_accuracy = correct_total / num_samples_total
        val_loss, val_acc = evaluate(net, criterion, val_loader, device)

        if scheduler is not None:
            # decay the learning rate
            if type(scheduler) == torch.optim.lr_scheduler.ReduceLROnPlateau:
                # here, you can potentially track on either of the two metrics: val loss or val accuracy
                scheduler.step(val_loss)
            else:
                scheduler.step()

        print('Epoch: %d' % (epoch + 1))
        print('Training Loss: %.3f , Training Accuracy:%.3f => Validation Loss: %.3f, Validation Accuracy: %.3f ' % (running_loss, train_accuracy, val_loss, val_acc))

        losses.append(running_loss)  # list containing the training losses
        val_losses.append(val_loss)  # list containing the validation losses
        accuracy.append(train_accuracy)  # list containing the training accuracy
        val_accuracy.append(val_acc)  # list containing the validation accuracy

        if early_stopping is not None:
            # early stopping needs to check the validation loss if it decreases, and saves checkpoint of current best model
            early_stopping(val_loss, net)

            if early_stopping.early_stop:
                print("Early Stopping...")
                break

    print('Finished Training')
    return accuracy, val_accuracy, losses, val_losses


#
# FOR METRICS AND VALIDATION EVALUATION
#


def get_accuracy(predicted, labels):
    "Function to obtain the accuracy between the predicted labels and true labels"
    batch_len, correct = 0, 0
    batch_len = labels.size(0)
    correct = (predicted == labels).sum().item()
    return batch_len, correct


def evaluate(model, criterion, val_loader, device):
    "Function to perform evaluation for the respective models"
    losses = 0
    num_samples_total = 0
    correct_total = 0
    model.eval()
    for inputs, labels in val_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        out = model(inputs)
        _, predicted = torch.max(out, 1)
        loss = criterion(out, labels)
        losses += loss.item()
        b_len, corr = get_accuracy(predicted, labels)
        num_samples_total += b_len
        correct_total += corr
    accuracy = correct_total / num_samples_total
    losses = losses / len(val_loader)
    return losses, accuracy


class EarlyStopping:
    "Function to stop the training early, if the validation loss doesn't improve after a predefined patience."
    def __init__(self, checkpoint_path, patience=5, verbose=False, delta=0):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta
        self.model_checkpoint_path = checkpoint_path

    def __call__(self, val_loss, model):

        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            print(f'EarlyStopping Counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        '''Saves model when validation loss decrease.'''
        if self.verbose:
            print(f'Validation Loss Decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), self.model_checkpoint_path)
        self.val_loss_min = val_loss


def get_metric_plots(train_metric, val_metric, metric_name: str = None):
    "Function to plot the respective metric (accuracy or loss) for training and validation"
    plt.figure()
    plt.plot(train_metric)
    plt.plot(val_metric)
    if metric_name is not None:
        plt.title(metric_name.upper())
        plt.ylabel(metric_name)
    else:
        print("It's expected to pass the name of metrics in the `metric_name` argument. (e.g. if you are visualizing train and val accuracies, please provide `metric_name=\"accuracy\"`, etc.)")
    plt.xlabel('epochs')
    plt.legend(['train', 'validation'], loc='upper left')


#
# FOR TEST EVALUATION
#


def test_evaluation(net, test_loader, device):
    "Function to get the test evaluation on the respective model"
    correct = 0
    total = 0
    net.eval()

    with torch.no_grad():
        for data in test_loader:
            images, labels = data
            images, labels = images.to(device), labels.to(device)
            outputs = net(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    return correct, total


def get_confusion_matrix(classes, test_loader, net, device, view_cm=False):
    confusion_matrix = torch.zeros(len(classes), len(classes))
    with torch.no_grad():
        for data in test_loader:
            images, labels = data
            images, labels = images.to(device), labels.to(device)
            outputs = net(images)
            _, predicted = torch.max(outputs, 1)
            for t, p in zip(labels.view(-1), predicted.view(-1)):
                confusion_matrix[t.long(), p.long()] += 1
    if view_cm:
        print(confusion_matrix, "\n")
    cm = confusion_matrix.numpy()
    return cm


def get_metrics_from_confusion_matrix(cm, chosen_index):
    total = np.sum(cm)
    idx = chosen_index
    recall = cm[idx][idx] / np.sum(cm[:, idx])
    precision = cm[idx][idx] / np.sum(cm[idx])
    accuracy = cm[idx][idx] + (total - np.sum(cm[idx]) - np.sum(cm[:, idx]) + cm[idx][idx])
    return recall, precision, accuracy / total


def check_precision_recall_accuracy(cm, all_classes):
    "Function to getting the "
    for i, _class in enumerate(all_classes):
        recall, precision, accuracy = get_metrics_from_confusion_matrix(cm, i)
        print(f"{_class} - recall : ", recall, " precision : ", precision, " accuracy : ", accuracy)


def visualize_confusion_matrix(cm, classes, correct, total):
    fig, ax = plt.subplots(figsize=(11, 11))
    sns.heatmap(cm / (cm.astype("float").sum(axis=1) + 1e-9), annot=True, ax=ax)

    # labels, title and ticks for plotting the confusion matrix
    ax.set_xlabel('Predicted', size=25)
    ax.set_ylabel('True', size=25)
    ax.set_title('Confusion Matrix', size=25)
    ax.xaxis.set_ticklabels(classes, size=5)
    ax.yaxis.set_ticklabels(classes, size=5)
    print(correct / total)


#
# FUNCTION TO GET THE INFERENCE FROM THE MODEL ON THE HOLD-OUT TEST SET
#


# This function takes a model, runs prediction for all images in the 'unknown' folder with it
# and then stores the predictions in a csv file so that predictions can be mapped to filenames.
# You can use this function to obtain the predictions from your best model.
def predict_unknown(model, height, width, norm_mean, norm_std, unknown_dir, device="cpu", filename=None):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((height, width)),
        transforms.Normalize(norm_mean, norm_std)
    ])

    image_names = []
    predictions = []

    with torch.no_grad():
        model.eval()
        images = glob(os.path.join(unknown_dir, "*.jpg"))
        for image_path in tqdm(images):
            image_name = os.path.basename(image_path)
            image_names.append(image_name)

            image = imageio.imread(image_path)
            input_ = transform(image).to(device)[None]
            prediction = model(input_).argmax(dim=1)
            predictions.append(prediction.cpu().numpy().item())


    assert len(predictions) == len(image_names)
    result = pd.DataFrame.from_dict({
        "image_name": image_names, "prediction": predictions
    })

    if filename is None:
        filename = "unknown_predictions.csv"

    result.to_csv(filename, index=False)
