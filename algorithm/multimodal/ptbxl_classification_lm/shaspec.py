from ...fedbase import BasicServer, BasicClient
import utils.system_simulator as ss
from utils import fmodule
import copy
import collections
import utils.fflow as flw
import os
import torch
import numpy as np
import wandb
import random

class Server(BasicServer):
    def __init__(self, option, model, clients, test_data = None):
        super(Server, self).__init__(option, model, clients, test_data)
        self.n_leads = 12
        self.list_testing_leads = [
            [2, 6, 10],                         #1
            [1, 2, 6, 10, 11],                  #2
            [1, 2, 6, 9, 10],                   #3
            [2, 4, 5, 9, 10, 11],               #4
            [2, 3, 4, 5, 6, 7, 9, 10, 11],      #5
            [2, 4, 5, 6, 7, 8, 9, 11],          #6
            [0, 1, 2, 4, 5, 6, 7, 8, 9, 11]     #7
        ]
        self.checkpoints_dir = os.path.join('fedtask', option['task'], 'checkpoints')
        os.makedirs(self.checkpoints_dir, exist_ok=True)

    def run(self):
        """
        Start the federated learning symtem where the global model is trained iteratively.
        """
        flw.logger.time_start('Total Time Cost')
        for round in range(1, self.num_rounds+1):
            self.current_round = round
            ss.clock.step()
            # using logger to evaluate the model
            flw.logger.info("--------------Round {}--------------".format(round))
            flw.logger.time_start('Time Cost')
            if flw.logger.check_if_log(round, self.eval_interval) and round > 1:
                flw.logger.time_start('Eval Time Cost')
                flw.logger.log_once()
                flw.logger.time_end('Eval Time Cost')
            # check if early stopping
            if flw.logger.early_stop(): break
            # federated train
            self.iterate()
            # decay learning rate
            self.global_lr_scheduler(round)
            flw.logger.time_end('Time Cost')
        flw.logger.info("--------------Final Evaluation--------------")
        flw.logger.time_start('Eval Time Cost')
        flw.logger.log_once()
        flw.logger.time_end('Eval Time Cost')
        flw.logger.info("=================End==================")
        flw.logger.time_end('Total Time Cost')
        # save results as .json file
        flw.logger.save_output_as_json()
        return

    def save_checkpoints(self):
        print("Saving global model checkpoints!")
        if not os.path.exists(os.path.join(self.checkpoints_dir, 'global-model')):
            os.makedirs(os.path.join(self.checkpoints_dir, 'global-model'), exist_ok=True)
        # torch.save(self.model.feature_extractors.state_dict(), os.path.join(self.checkpoints_dir, 'global-model', 'feature_extractor.pt'))
        # torch.save(self.model.branchallleads_classifier.state_dict(), os.path.join(self.checkpoints_dir, 'global-model', 'classifier.pt'))
        torch.save(self.model.state_dict(), os.path.join(self.checkpoints_dir, 'global-model', 'model.pt'))

        # if flw.logger.output['test_2_loss'][-1] == min(flw.logger.output['test_2_loss']):
        #     os.makedirs(os.path.join(self.checkpoints_dir, 'missing-modal'), exist_ok=True)
        #     print("Saving missing-modal model checkpoints!")
        #     torch.save(self.model.branch2leads.state_dict(), os.path.join(self.checkpoints_dir, 'missing-modal', 'feature_extractor.pt'))
        #     torch.save(self.model.branch2leads_classifier.state_dict(), os.path.join(self.checkpoints_dir, 'missing-modal', 'classifier.pt'))

    def load_checkpoints(self):
        if os.path.exists(os.path.join(self.checkpoints_dir, 'global-model')):
            print("Loading global model checkpoints!")
            self.model.load_state_dict(torch.load(os.path.join(self.checkpoints_dir, 'global-model', 'model.pt')))


    def iterate(self):
        """
        The standard iteration of each federated round that contains three
        necessary procedure in FL: client selection, communication and model aggregation.
        :param
            t: the number of current round
        """
        # sample clients: MD sampling as default
        self.selected_clients = self.sample()
        # training
        conmmunitcation_result = self.communicate(self.selected_clients)
        models = conmmunitcation_result['model']
        modalities_list = conmmunitcation_result['modalities']
        # if wandb.run.resumed:
        #     self.load_checkpoints()
        self.model = self.aggregate(models, modalities_list)
        # self.save_checkpoints()
        return

    @torch.no_grad()
    def aggregate(self, models: list, modalities_list: list):
        print("Calculating clients' aggregated models ...")
        n_models = len(models)
        new_model = copy.deepcopy(self.model)
        
        # modality prototypes
        for m in range(self.n_leads):
            new_model.modality_prototypes[m] = fmodule._model_average(
                [self.clients[self.selected_clients[k]].local_model.modality_prototypes[m] for k in range(n_models)])
            
        # modality projectors
        for m in range(self.n_leads):
            new_model.modality_projectors[m] = fmodule._model_average(
                [self.clients[self.selected_clients[k]].local_model.modality_projectors[m] for k in range(n_models)])
            
        # global align classifier
        new_model.global_align_classifier = fmodule._model_average(
            [self.clients[self.selected_clients[k]].local_model.global_align_classifier for k in range(n_models)])
        
        # global shared extractor
        new_model.global_shared_extractor = fmodule._model_average(
            [self.clients[self.selected_clients[k]].local_model.global_shared_extractor for k in range(n_models)])
        
        # classifier
        new_model.classifier = fmodule._model_average(
            [self.clients[self.selected_clients[k]].local_model.classifier for k in range(n_models)])
                
        # NOTE: local_align_classifier and local_modality_classifier are unique for each client -> Not aggregate
            
        return new_model
    
    def test(self, model=None):
        """
        Evaluate the model on the test dataset owned by the server.
        :param
            model: the model need to be evaluated
        :return:
            metrics: specified by the task during running time (e.g. metric = [mean_accuracy, mean_loss] when the task is classification)
        """
        # return dict()
        if model is None: model=self.model
        if self.test_data:
            return self.calculator.server_test(
                model=model,
                dataset=self.test_data,
                batch_size=self.option['test_batch_size'],
                leads=self.list_testing_leads
            )
        else:
            return None

    def test_on_clients(self, dataflag='train'):
        """
        Validate accuracies and losses on clients' local datasets
        :param
            dataflag: choose train data or valid data to evaluate
        :return
            metrics: a dict contains the lists of each metric_value of the clients
        """
        all_metrics = collections.defaultdict(list)
        for client_id in self.selected_clients:
            c = self.clients[client_id]
            client_metrics = c.test(c.local_model, dataflag)
            for met_name, met_val in client_metrics.items():
                all_metrics[met_name].append(met_val)
        return all_metrics


class Client(BasicClient):
    def __init__(self, option, modalities, name='', train_data=None, valid_data=None):
        super(Client, self).__init__(option, name, train_data, valid_data)
        self.n_leads = 12
        self.pm = option['pm']
        self.ps = option['ps']
        self.train_data = train_data
        self.train_data.local_missing_setup(modalities, self.ps, self.pm)   # local missing dataset setting
        self.modalities = modalities
        self.local_cls_lambda = option['local_cls_lambda']
        self.local_aln_lambda = option['local_aln_lambda']
        self.global_aln_lambda = option['global_aln_lambda']
        self.local_model = fmodule.CltModel()
        self.local_model.set_local_leads(modalities)

    def pack(self, model):
        """
        Packing the package to be send to the server. The operations of compression
        of encryption of the package should be done here.
        :param
            model: the locally trained model
        :return
            package: a dict that contains the necessary information for the server
        """
        return {
            "model" : model,
            "modalities": self.modalities
        }

    def reply(self, svr_pkg):
        """
        Reply to server with the transmitted package.
        The whole local procedure should be planned here.
        The standard form consists of three procedure:
        unpacking the server_package to obtain the global model,
        training the global model, and finally packing the updated
        model into client_package.
        :param
            svr_pkg: the package received from the server
        :return:
            client_pkg: the package to be send to the server
        """
        model = self.unpack(svr_pkg)
        # if self.local_model is None:
        #     self.local_model = copy.deepcopy(model)
        # if self.agg_model is None:
        #     self.agg_model = copy.deepcopy(model)
        self.train(self.local_model)
        cpkg = self.pack(self.local_model)
        return cpkg

    @ss.with_completeness
    @fmodule.with_multi_gpus
    def train(self, model):
        """
        Standard local training procedure. Train the transmitted model with local training dataset.
        :param
            model: the global model
        :return
        """
        model.train()
        optimizer = self.calculator.get_optimizer(
            model=model,
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
            momentum=self.momentum
        )
        for iter in range(self.num_steps):
            # get a batch of data
            batch_data = self.get_batch_data()
            if batch_data[-1].shape[0] == 1:
                continue
            model.zero_grad()
            # calculate the loss of the model on batched dataset through task-specified calculator
            loss_list, downstream_loss, outputs = self.calculator.train_one_step(
                model=model,
                data=batch_data,
                leads=self.modalities,
            )['loss']
            loss = loss_list['downstream'] + loss_list['local_cls'] * self.local_cls_lambda + loss_list['local_aln'] * self.local_aln_lambda + \
                loss_list['global_aln'] * self.global_aln_lambda
            loss.backward()
            optimizer.step()
        return 

    @fmodule.with_multi_gpus
    def test(self, model, dataflag='train'):
        """
        Evaluate the model with local data (e.g. training data or validating data).
        :param
            model:
            dataflag: choose the dataset to be evaluated on
        :return:
            metric: specified by the task during running time (e.g. metric = [mean_accuracy, mean_loss] when the task is classification)
        """
        if dataflag == "train":
            dataset = self.train_data
        elif dataflag == "valid":
            dataset = self.valid_data
        return self.calculator.test(
            model=model,
            dataset=dataset,
            leads=self.modalities
        )