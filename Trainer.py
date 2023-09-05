from DataPrep import DataPrep 
from DB import DB
from EarlyStopping import EarlyStopping
from torchvision import transforms
from torch.utils.data.sampler import SubsetRandomSampler
from ray import train
import torch
import torch.nn as nn
import numpy as np
from hashlib import sha256


# This module is in charge of handling the curriculum aspect of the training process.
# Requirements:
# - Should be able to specify type of training (curriculum or not).
# - Should be able to handle the curriculum dynamically.
class Trainer:
    # @param datasets: A list of paths to the datasets
    # @param model: The model to be trained
    def __init__(self, data_dirs, model, device, seed = 1):
        self.data_dirs = data_dirs
        self.model = model
        self.device = device
        self.random_seed = seed
        self.early_stopping = EarlyStopping()
        self.valid_size = 0.1
        self.data_prep = DataPrep()
        self._load_data()
        self._prepare_data()
    
    def train(self, options, tune, wandb, report_logs, should_tune):
        # Options have all the hyperparameters
        model = self.model.to(self.device)
        criterion = options['criterion']
        optimizer = torch.optim.SGD(model.parameters(), lr=options['learning_rate'], weight_decay=options['weight_decay'], momentum=options['momentum'])

        for i in range(len(self.data_dirs)):
            epochs = self.data_dirs[i]['epochs']
            print('Training on dataset: ' + self.data_dirs[i]['dataset'])

            if options['curriculum']:
                self._bootstrap_data(self.train_sets[i], criterion, options['batch_size'])

            train_loader = torch.utils.data.DataLoader(self.train_sets[i], batch_size=options['batch_size'])
            valid_loader = torch.utils.data.DataLoader(
                self.valid_sets[i], batch_size=options['batch_size'], sampler=self.valid_samplers[i]
                )

            total_step = len(train_loader)
            for epoch in range(epochs):
                for i, (images, labels) in enumerate(train_loader):  
                    # print(labels)
                    # Move tensors to the configured device
                    images = images.to(self.device)
                    labels = labels.to(self.device)
                    
                    # Forward pass
                    outputs = model(images)
                    loss = criterion(outputs, labels)
                    
                    # Backward and optimize
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                self.early_stopping.save_checkpoint(model, optimizer, epoch)

                print(loss.item())
                curr_loss = loss.item()
                loss, acc = self._validate(valid_loader, criterion)
                # print ('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}' 
                #             .format(epoch+1, epochs, i+1, total_step, curr_loss))

                if should_tune:
                    tune.report({"validation_loss": loss, "training_loss": curr_loss, "accuracy": acc})

                if report_logs:
                    wandb.log({"validation_loss": loss, "training_loss": curr_loss, "epoch": epoch, "accuracy": acc})

                # Check if converged

            self._replace_classifier(10)

    def test(self):
        with torch.no_grad():
            correct = 0
            total = 0
            for images, labels in test_loader:
                images = images.to(self.device)
                labels = labels.to(self.device)
                outputs = self.model(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                del images, labels, outputs

            print('Accuracy of the network on the {} test images: {} %'.format(10000, 100 * correct / total)) 

    def _validate(self, valid_loader, criterion):
        # Validation
        with torch.no_grad():
            correct = 0
            total = 0
            for images, labels in valid_loader:
                images = images.to(self.device)
                labels = labels.to(self.device)
                outputs = self.model(images)
                _, predicted = torch.max(outputs.data, 1)
                loss = criterion(outputs, labels)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                del images, labels, outputs
            # print('Accuracy of the network on the {} validation images: {} %'.format(total, 100 * correct / total))
        return loss.item(), 100 * correct / total
    
    def _load_data(self):
        print('loading the data')
        self.train_sets = []
        self.valid_sets = []
        self.test_sets = []
        for dir in self.data_dirs:
            train_set, valid_set = self.data_prep.get_datasets(data_dir=dir['dataset'], model=self.model)
            # test_set = DataPrep.get_test_set(data_dir=dir['dataset'])
            self.train_sets.append(train_set)
            self.valid_sets.append(valid_set)
            # self.test_sets.append(test_set)

    def _prepare_data(self, augment = False):
        # Calculate mean and std on training set and use that throughout the training and testing process
        print('preparing the data')
        self.train_samplers = []
        self.valid_samplers = []
        for i in range(len(self.data_dirs)):
            # TODO: Add the mean and std to the DB and load them here if they exist
            # print(sha256(self.data_dirs[i]['datasets'].encode('utf-8')).hexdigest())
            result = self.data_prep.compute_mean_std(self.train_sets[i])
            normalize = transforms.Normalize(
                mean= result['mean'],
                std= result['std'],
            )
            size = 227
       
            test_transform = transforms.Compose([
                transforms.Resize((size,size)),
                transforms.ToTensor(),
                normalize,
            ])
            valid_transform = transforms.Compose([
                transforms.Resize((size,size)),
                transforms.ToTensor(),
                normalize,
            ])
            if augment:
                train_transform = transforms.Compose([
                    transforms.RandomCrop(32, padding=4),
                    transforms.RandomHorizontalFlip(),
                    transforms.Resize((size,size)),
                    transforms.ToTensor(),
                    normalize,
                ])
            else:
                train_transform = transforms.Compose([
                    transforms.Resize((size,size)),
                    transforms.ToTensor(),
                    normalize,
                ])

            self.train_sets[i].transform = train_transform
            self.valid_sets[i].transform = valid_transform
            # self.test_sets[i].transform = test_transform

            self._split_train_valid(self.train_sets[i])
        
    def _replace_classifier(self, num_classes):
        self.model.to('cpu')
        self.model.fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(9216, 4096),
            nn.ReLU())
        self.model.fc1 = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(4096, 4096),
            nn.ReLU())
        self.model.fc2= nn.Sequential(
            nn.Linear(4096, num_classes))
        self.model.to(self.device)

        # Add the layers that are responsible for classification (with the specefic number of classes)
        # Also play around with adjusting the learning rate of these layers
        # AlexNet: fc, fc1, fc2
        # ResNet50: fc

    def _split_train_valid(self, train_set, shuffle=True):
        num_train = len(train_set)
        indices = list(range(num_train))
        split = int(np.floor(self.valid_size * num_train))

        if shuffle:
            np.random.seed(self.random_seed)
            np.random.shuffle(indices)

        train_idx, valid_idx = indices[split:], indices[:split]
        self.train_samplers.append(SubsetRandomSampler(train_idx))
        self.valid_samplers.append(SubsetRandomSampler(valid_idx))

    def _bootstrap_data(self, data_set, criterion, batches):
        # TODO: You can access the class, so the solution is to compute the error values and then reorder the data items in the class
        data_loader = torch.utils.data.DataLoader(data_set)
        self.model.eval()
        total = 0
        correct = 0
        predictions = []
        for image, label in data_loader:
            with torch.no_grad():
                image = image.to(self.device)
                label = label.to(self.device)
                outputs = self.model(image)
                _, predicted = torch.max(outputs.data, 1)
                predictions.append((criterion(outputs, label).item(),total,label))
                total += 1
                correct += (predicted == label).sum().item()
        self.model.train()
        sorted_predictions = self.sort_by_error(predictions)
        print(sorted_predictions)
        data_set.reorder(sorted_predictions)

    def sort_by_error(self, predictions):
        return sorted(predictions, key=lambda x: x[0])

    def _save_model(self):
        pass

    def _load_model(self):
        pass