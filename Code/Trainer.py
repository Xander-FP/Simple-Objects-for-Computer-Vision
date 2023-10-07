from DataPrep import DataPrep 
from EarlyStopping import EarlyStopping
from torchvision import transforms
from ray import train
from Scheduler import BabyStep, RootP, Scheduler
from torch.utils.data.sampler import SubsetRandomSampler
import torch
import torch.nn as nn
import numpy as np
import os
import time
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
        size = len(data_dirs)

        self.train_sets = [None] * size
        self.valid_sets = [None] * size
        self.test_sets = [None] * size
        self.train_samplers = [None] * size
        self.valid_samplers = [None] * size
        self._load_data()
    
    def start(self, options, tune, wandb):
        train_options = options
        model = self.model.to(self.device)
        # maybe not needed
        # max_epochs = options['epochs']

        for i in range(len(self.data_dirs)):
            data_path = self.data_dirs[i]['path']
            print('Training on dataset: ' + data_path)
            self._prepare_data(i, options['architecture'])

            if options['curriculum']:
                self._bootstrap_data(self.train_sets[i], options['criterion'], options['batch_size'], self.data_dirs[i]['name'])

            data_size = len(self.train_sets[i]) 

            if options['scheduler'] == 'R':
                scheduler = RootP(data_size=data_size, max_epochs=options['epochs'], start=0.2)
            elif options['scheduler'] == 'B':
                scheduler = BabyStep(data_size=data_size, num_buckets=10, max_epochs=options['epochs'])
            else:
                scheduler = Scheduler(data_size=data_size)

            train_idx, valid_idx = scheduler.get_initial_indexes()
            train_loader = torch.utils.data.DataLoader(self.train_sets[i], batch_size=options['batch_size'], sampler=SubsetRandomSampler(train_idx))
            valid_loader = torch.utils.data.DataLoader(self.valid_sets[i], batch_size=options['batch_size'], sampler=SubsetRandomSampler(valid_idx))

            train_options['scheduler_object'] = scheduler

            # TODO: Add augmentation here in a seperate function
            val_loss = self._train(
                model = model,
                train_loader = train_loader,
                valid_loader = valid_loader,
                tune = tune,
                wandb = wandb,
                options = train_options
                )

            # min(self.early_stopping.history, key=lambda x: x['validation_loss'])
            print(self.early_stopping.history)
            if i != len(self.data_dirs) - 1:
                self._replace_classifier(self.data_dirs[i+1]['classes'])
            else:
                return val_loss
            # return the best and its epoch 
            # return val_loss

    def _train(self, model, tune, wandb, train_loader, valid_loader, options):
        # SETUP PHASE
        if options['opt'] == 'sgd':
            print('SGD')
            optimizer = torch.optim.SGD(model.parameters(), lr=options['learning_rate'], weight_decay=options['weight_decay'], momentum=options['momentum'])
        else:
            print('ADAM')
            optimizer = torch.optim.Adam(model.parameters(), lr=options['learning_rate'], weight_decay=options['weight_decay'])    
        scheduler = options['scheduler_object']
        criterion = options['criterion']
        max_epochs = options['epochs']
        model.train()
        total_step = len(train_loader)
        file = open(options['architecture'] + str(time.time()) + '.txt', 'a')
        # TRAINING PHASE
        while not scheduler.converged:
            print('Training started')
            for epoch in range(max_epochs):
                for i, (images, labels) in enumerate(train_loader):  
                    images = images.to(self.device)
                    labels = labels.to(self.device)
                    
                    # Forward pass
                    outputs = model(images)
                    loss = criterion(outputs, labels)
                    
                    # Backward and optimize
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                train_loss = loss.item()
                valid_loss, acc = self._validate(valid_loader, criterion)
                print ('Epoch [{}/{}], Train_Loss: {:.4f}, Valid_Loss: {:.4f}' 
                            .format(epoch+1, max_epochs, train_loss, valid_loss))
                file.write('Epoch [{epoch+1/{}], Train_Loss: {:.4f}, Valid_Loss: {:.4f}\n')
                
                # self.early_stopping.save_checkpoint(model, optimizer, epoch, valid_loss)
                converged = scheduler.adjust_available_data(self.early_stopping, train_loss, valid_loss)

                if converged:
                    break

                # if options['should_tune']:
                    # tune.report({"validation_loss": valid_loss, "training_loss": train_loss, "accuracy": acc})

                if options['report_logs']:
                    wandb.log({"validation_loss": valid_loss, "training_loss": train_loss, "epoch": epoch, "accuracy": acc})
        print('Training finished')
        if options['should_tune']:
            if not os.path.exists('h_results'):
                os.makedirs('h_results')
            path = 'h_results/results.txt' 
            file = open(path, 'a')
            options['validation_loss'] = valid_loss
            options['training_loss'] = train_loss
            options['accuracy'] = acc
            file.write(str(options) + '\n\n')
        return valid_loss
        

    def _test(self):
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
            print('Accuracy of the network on the {} validation images: {} %'.format(total, 100 * correct / total))
        return loss.item(), 100 * correct / total
    
    def _load_data(self):
        print('loading the data')
        i = 0
        for dir in self.data_dirs:
            train_set, valid_set = self.data_prep.get_datasets(data_dir=dir['path'], model=self.model)
            # test_set = DataPrep.get_test_set(data_dir=dir['path'])
            self.train_sets[i] = train_set
            self.valid_sets[i] = valid_set
            # self.test_sets[i] = test_set
            i = i + 1

    def _prepare_data(self, i, architecture):
        # Calculate mean and std on training set and use that throughout the training and testing process
        print('Preparing the data: Adding transforms')
        # TODO: Add the mean and std to the DB and load them here if they exist
        # print(sha256(self.data_dirs[i]['paths'].encode('utf-8')).hexdigest())
        result = self.data_prep.compute_mean_std(self.train_sets[i])
        normalize = transforms.Normalize(
            mean= result['mean'],
            std= result['std'],
        )
        if architecture == 'ResNet':
            size = 224
        else:
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
        train_transform = transforms.Compose([
            transforms.Resize((size,size)),
            transforms.ToTensor(),
            normalize,
        ])

        self.train_sets[i].transform = train_transform
        self.valid_sets[i].transform = valid_transform
        # self.test_sets[i].transform = test_transform
        
    def _replace_classifier(self, num_classes):
        self.early_stopping.reset()

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

    def _bootstrap_data(self, data_set, criterion, batches, name):
        print('Ordering data')
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
                predictions.append((criterion(outputs, label).item(),total))
                total += 1
                correct += (predicted == label).sum().item()
        self.model.train()
        sorted_predictions = self.sort_by_error(predictions)

        # Write the sorted predictions to a file
        path = 'predictions/' + name + str(time.time()) +'.txt'
        if not os.path.exists('predictions'):
            os.makedirs('predictions')
        if not os.path.exists(path):
            np.savetxt(path, sorted_predictions, fmt='%i')
        data_set.reorder(sorted_predictions)
        print('data ordered')

    def sort_by_error(self, predictions):
        return sorted(predictions, key=lambda x: x[0])
    
    def _augment_data(self):
        train_transform = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.Resize((size,size)),
            transforms.ToTensor(),
            normalize,
        ])
