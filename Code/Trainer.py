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
    def __init__(self, data_dirs, model, device, dataset_name):
        self.data_dirs = data_dirs
        self.model = model
        self.device = device
        self.early_stopping = EarlyStopping()
        self.valid_size = 0.1
        self.data_prep = DataPrep()
        size = len(data_dirs)

        self.train_sets = [None] * size
        self.valid_sets = [None] * size
        self.test_sets = [None] * size
        self.train_samplers = [None] * size
        self.valid_samplers = [None] * size
        self._load_data(dataset_name)

    def test(self, model, batch_size, criterion, do_regression):
        test_loader = torch.utils.data.DataLoader(self.test_sets[-1], batch_size=batch_size, shuffle=True)
        file = open('test_results' + str(time.time()) + '.csv', 'w')
        file.write('ID, extent\n')

        with torch.no_grad():
            model.eval()
            for images, labels in test_loader:
                images = images.to(self.device)
                outputs = model(images)
                if do_regression:
                    predicted = torch.mul(outputs,100).to(torch.int32)
                else:
                    _, predicted = torch.max(outputs.data, 1)
                for i in range(len(outputs)):
                    file.write(str(labels[i]) + ',' + str(predicted[i].item()) + '\n')
                del images, labels, outputs
            print('Done with test set') 
    
    def start(self, options, tune, wandb):
        train_options = options
        model = self.model.to(self.device)
        folder_path = 'main_results'
        restored = False
        options['start_epoch'] = 0

        if options['should_restore']:
            print('Restoring from epoch: ' + str(options['new_epoch']) + ' in checkpoint: ')
            restored = True
            self.early_stopping.load_checkpoint(model, None, options['new_epoch'] - 1, restore = True)
            options['start_epoch'] = options['new_epoch']
        elif os.path.exists('checkpointsRestore'):
            print('Restoring from checkpoint:')
            folder_path += 'Restore'
        
        if not options['should_tune']:
            self.early_stopping.name = options['architecture'] + str(time.time())
            if not os.path.exists(folder_path):
                os.makedirs(folder_path)
            results_file = open(os.path.join(folder_path, options['architecture'] + str(time.time()) + '.txt'), 'a')
        else:
            results_file = None

        for i in range(len(self.data_dirs)):
            if os.path.exists('checkpointsRestore') and not restored:
                print('Loading model from checkpoint')
                self.early_stopping.load_checkpoint(model, None, options['epochs'] - 1, restore = True)
                self._replace_classifier(model, self.data_dirs[i + 1]['classes'], options['architecture'])
                restored = True
                continue

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

            val_loss = self._train(
                model = model,
                train_loader = train_loader,
                valid_loader = valid_loader,
                tune = tune,
                wandb = wandb,
                options = train_options,
                results_file = results_file
                )
            
            options['start_epoch'] = 0

            min_res = min(self.early_stopping.history, key=lambda x: x['validation_loss'])
            self.early_stopping.load_checkpoint(model, None, min_res['epoch'])
            print('Loading epoch: ' + str(min_res['epoch']))
            if i != len(self.data_dirs) - 1:
                self._replace_classifier(model, self.data_dirs[i+1]['classes'], options['architecture'])
            else:
                return val_loss, model, results_file

    def _train(self, model, tune, wandb, train_loader, valid_loader, options, results_file):
        print('Current time: ', time.strftime("%H:%M:%S", time.localtime()))
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
        start_epoch = options['start_epoch']
        model.train()
        # TRAINING PHASE
        while not scheduler.converged:
            print('Training started')
            for epoch in range(start_epoch, max_epochs):
                for i, (images, labels) in enumerate(train_loader):  
                    images = images.to(self.device)
                    labels = labels.to(self.device)
                    
                    if options['regression']:
                        labels = labels.to(torch.float32)
                        outputs = torch.round(
                            torch.mul(model(images).view(-1), 100)
                        )
                        loss = torch.sqrt(criterion(outputs, labels))
                    else:
                        outputs = model(images)
                        loss = criterion(outputs, labels)
                    # Backward and optimize
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                train_loss = loss.item()
                valid_loss, acc = self._validate(model, valid_loader, criterion, options['regression'])
                print ('Epoch [{}/{}], Train_Loss: {:.4f}, Valid_Loss: {:.4f}' 
                            .format(epoch, max_epochs, train_loss, valid_loss))
                print('Current time: ', time.strftime("%H:%M:%S", time.localtime()))
                if not options['should_tune']:
                    results_file.write('{Epoch ' + str(epoch) + ', Train_Loss: ' + str(train_loss) + ', Valid_Loss: ' + str(valid_loss) + ', Accuracy: ' + str(acc) + '},\n')
                self.early_stopping.save_checkpoint(model, optimizer, epoch, valid_loss)
                converged = scheduler.adjust_available_data(self.early_stopping, train_loss, valid_loss)

                if converged:
                    break

                if options['report_logs']:
                    wandb.log({"validation_loss": valid_loss, "training_loss": train_loss, "epoch": epoch, "accuracy": acc})
        print('Training finished')
        if not options['should_tune']:
            results_file.write('\n{******************************Training finished**********************************},\n')
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
        
    def _validate(self, model, valid_loader, criterion, do_regression):
        # Validation
        with torch.no_grad():
            correct = 0
            total = 0
            for images, labels in valid_loader:
                images = images.to(self.device)
                labels = labels.to(self.device)
                outputs = self.model(images)
                if do_regression:
                    labels = labels.to(torch.float32)
                    predicted = torch.round(
                        torch.mul(outputs.view(-1), 100)
                    )
                    loss = torch.sqrt(criterion(predicted, labels))
                else:
                    criterion2 = nn.MSELoss()
                    labels2 = labels.to(torch.float32)
                    predicted2 = torch.round(
                                torch.mul(outputs, 100)
                            )
                    _, predicted2 = torch.max(predicted2.data, 1)
                    loss2 = torch.sqrt(criterion2(predicted2, labels2))
                    _, predicted = torch.max(outputs.data, 1)
                    loss = criterion(outputs, labels)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                del images, labels, outputs
            print('Accuracy of the network on the {} validation images: {} %'.format(total, 100 * correct / total))
            if not do_regression:
                print('RMSE: ' + str(loss2.item()))
        return loss.item(), 100 * correct / total
    
    def _load_data(self, dataset_name):
        print('loading the data')
        # Set the datasets
        i = 0
        for dir in self.data_dirs:
            train_set, valid_set = self.data_prep.get_datasets(data_dir=dir['path'], model=self.model)
            test_set = self.data_prep.get_test_datasets(data_dir=dir['path'])
            self.train_sets[i] = train_set
            self.valid_sets[i] = valid_set
            self.test_sets[i] = test_set
            i = i + 1

    def _prepare_data(self, i, architecture):
        # Calculate mean and std on training set and use that throughout the training and testing process
        print('Preparing the data: Adding transforms')
        # TODO: Add the mean and std to the DB and load them here if they exist
        # print(sha256(self.data_dirs[i]['paths'].encode('utf-8')).hexdigest())
        # result = self.data_prep.compute_mean_std(self.train_sets[i])
        # print(result)
        normalize = transforms.Normalize(
            mean = [0.4467916627098922, 0.42175631382363743, 0.3320200420338127], 
            std = [0.2626697991603522, 0.2694901367915744, 0.30393761999531593]
        )
        if architecture == 'ResNet':
            size = 224
        else:
            size = 280
    
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
        if self.test_sets[i] is not None:
            self.test_sets[i].transform = test_transform
        
    def _replace_classifier(self, model, num_classes, architecture):
        self.early_stopping.reset()
        print('Replacing the classifier')
        print(self.device)
        model.to('cpu')
        if architecture == 'ResNet':
            model.fc = nn.Linear(512, num_classes)
        else:
            model.fc = nn.Sequential(
                nn.Dropout(0.5),
                nn.Linear(9216, 4096),
                nn.ReLU())
            model.fc1 = nn.Sequential(
                nn.Dropout(0.5),
                nn.Linear(4096, 4096),
                nn.ReLU())
            model.fc2= nn.Sequential(
                nn.Linear(4096, num_classes))
        model.to(self.device)

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
        # path = 'predictions/' + name + str(time.time()) +'.txt'
        # if not os.path.exists('predictions'):
        #     os.makedirs('predictions')
        # if not os.path.exists(path):
        #     np.savetxt(path, sorted_predictions, fmt='%i')
        data_set.reorder(sorted_predictions)
        print('data ordered')

    def sort_by_error(self, predictions):
        return sorted(predictions, key=lambda x: x[0])