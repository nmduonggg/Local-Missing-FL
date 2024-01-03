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
from tqdm import tqdm

class Server(BasicServer):
    def __init__(self, option, model, clients, test_data=None):
        super(Server, self).__init__(option, model, clients, test_data)
        self.n_leads=2
        self.list_testing_leads=[
            [0, 1]
        ]
        self.prompt_pools=None
        self.checkpoints_dir = os.path.join('fedtask', option['task'], 'checkpoints')
        os.makedirs(self.checkpoints_dir, exist_ok=True)
        self.model = fmodule.SvrModel()
        
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
        self.selected_clients = self.sample()
        # training
        communication_result = self.communicate(self.selected_clients)
        prompts = communication_result['prompts']
        prototypes = communication_result['prototypes']
        modalities_list = communication_result['modalities']    # modalities that client has
        self.prompt_pools = self.aggregate([prompts, prototypes], modalities_list)

        return
    
    def aggregate(self, model_components: list, modalities_list: list):
        
        n_models = len(model_components)
        modal_dict = dict()
        for m in range(self.n_leads):
            modal_dict[m] = list()
            for k in range(n_models):
                if m in modalities_list[k]:
                    modal_dict[m].append(k)
        self.modal_dict = modal_dict
                    
        
        prompts, prototypes = model_components
        n_models = len(prototypes)  # each model contains only 1 prototype (parameter list)
        
        modal_dict = dict()
        for m in range(self.n_leads):
            modal_dict[m] = list()
            for k in range(n_models):
                if m in modalities_list[k]:
                    modal_dict[m].append(k)
                    
        prompt_pools = []
        for m in range(self.n_leads):
            pr_length, _, pr_dim = prompts[0][m].prompt.size()
            pt_length, _, pt_dim = prototypes[0][m].prototype.size()
            assert (pr_dim==pt_dim)
            pool = torch.zeros(size=(len(modal_dict[m]), pr_length+pt_length, pr_dim))
            client_ids = []
            for i, k in enumerate(modal_dict[m]):
                prompt_list = [p for p in self.clients[self.selected_clients[k]].model.prompts[m].parameters()]
                prtt_list = [p for p in self.clients[self.selected_clients[k]].model.prototypes[m].parameters()]
                # print("Prompt len:", len(prompt_list))
                # print("Prtt len:", len(prtt_list))
                pool[i] = torch.cat([prompt_list[0].data.squeeze(1), prtt_list[0].data.squeeze(1)], dim=0)
                client_ids.append(self.selected_clients[k])
            prompt_pools.append([client_ids, pool])
        
        # prompt - prototype attn matrix
        A = torch.zeros(size=(self.n_leads, n_models, n_models))
        for m in range(self.n_leads):
            # no client has lead m in the current round, in this case, prompt pool of m = [0] -> atn score in inference stage = 0
            if len(modal_dict[m])==0: continue  
            prtts = torch.stack([
                torch.cat([
                    ui.data.view(-1) for ui in self.clients[self.selected_clients[k]].model.prototypes[m].parameters()
                ]) for k in modal_dict[m]
            ])  # K x total_num_of_prtt_params
            dim = prtts.shape[1]
            prtt_logits = prtts @ prtts.t() / np.sqrt(dim)
            
            params = torch.stack([
                torch.cat([
                    pi.data.view(-1) for pi in self.clients[self.selected_clients[k]].model.prompts[m].parameters()
                ]) for k in modal_dict[m]
            ])
            dim = params.shape[1]
            prm_logits = params @ params.t() / np.sqrt(dim) # n_clients x n_clients
            
            final_attn = torch.exp((prtt_logits + prm_logits)*0.5)
            for ik, k in enumerate(modal_dict[m]):
                for il, l in enumerate(modal_dict[m]):
                    A[m, k, l] = final_attn[ik, il]
            
        # personalized aggregate
        for m in range(self.n_leads):
            for k in modal_dict[m]:
                # prompt
                self.clients[self.selected_clients[k]].model.prompts[m] = fmodule._model_sum([
                    self.clients[self.selected_clients[l]].model.prompts[m] * (A[m, k, l] / torch.sum(A[m, k, :].sum(dim=-1))) for l in modal_dict[m]
                ])
                # prototype
                self.clients[self.selected_clients[k]].model.prototypes[m] = fmodule._model_sum([
                    self.clients[self.selected_clients[l]].model.prototypes[m] * (A[m, k, l] / torch.sum(A[m, k, :].sum(dim=-1))) for l in modal_dict[m]
                ])
        
        num_lead = {}
        for m in range(self.n_leads):
            for k in modal_dict[m]:
                num_lead[k] = num_lead.get(k, 0) + 1
        
        for k in range(n_models):        
            # classifier
            if num_lead[k]==2:
                self.clients[self.selected_clients[k]].model.classifier = fmodule._model_sum([
                    self.clients[self.selected_clients[l]].model.classifier * (A[:, k, l].sum() / torch.sum(A[:, k, :])) for l in modal_dict[m]
                ])
            
        # global classifier
        self.model.classifier = fmodule._model_average([
            self.clients[self.selected_clients[k]].model.classifier for k in range(n_models)
        ])
                
        return prompt_pools
    
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
            client_metrics = c.test(c.model, dataflag)
            for met_name, met_val in client_metrics.items():
                all_metrics[met_name].append(met_val)
        return all_metrics
    
    
    
    def pack(self, client_id):
        """
        Pack the necessary information for the client's local training.
        Any operations of compression or encryption should be done here.
        :param
            client_id: the id of the client to communicate with
        :return
            a dict that only contains the global model as default.
        """
        return_pool = [None, None]
        if self.prompt_pools is not None:  
            for m in range(self.n_leads):
                for i, k in enumerate(self.prompt_pools[m][0]): # client ids
                    if self.selected_clients[i]==client_id: 
                        return_pool[m] = self.prompt_pools[m][i]
                    
        return {
            "prompt_pools" : return_pool,
            "prompt_len": self.model.prompt_lengths[-1]
        }
            
    def test(self):
        """
        Evaluate the model on the test dataset owned by the server.
        :param
            model: the model need to be evaluated
        :return:
            metrics: specified by the task during running time (e.g. metric = [mean_accuracy, mean_loss] when the task is classification)
        """
        # return dict()
        prompt_pools = self.prompt_pools
        assert self.test_data is not None
        return self.calculator.server_test(
            model = self.model,
            prompt_pools=prompt_pools,
            dataset=self.test_data,
            batch_size=self.option['test_batch_size'],
            leads=self.list_testing_leads
        )
            
class Client(BasicClient):
    def __init__(self, option, modalities, name='', train_data=None, valid_data=None):
        super(Client, self).__init__(option, name, train_data, valid_data)
        self.n_leads=2
        self.pm=option['pm']
        self.ps=option['ps']
        self.modalities=modalities
        self.train_data=train_data
        self.valid_data=valid_data
        if len(modalities) > 1:
            self.train_data.local_missing_setup(modalities, self.ps, self.pm)
        self.model = fmodule.CltModel()
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
            "prompts": model.prompts,
            "prototypes": model.prototypes,
            "modalities": self.modalities
        }
        
    @ss.with_completeness
    @fmodule.with_multi_gpus
    def train(self, model):
        """
        Standard local training procedure. Train the transmitted model with local training dataset.
        :param
            model: the global model
        :return
        """
        
        # track number of parameters to make sure FM is frozen
        n_trainable = 0
        for n, v in model.named_parameters():
            if ('prompts' not in n) & ('prototypes' not in n):
                v.require_grad = False
            else: n_trainable += 1
        print("Number of trainable paramters in client model", self.id, "is:", n_trainable)
            
        # training step
        model.train()
        optimizer = self.calculator.get_optimizer(model, lr = self.learning_rate, weight_decay=self.weight_decay, momentum=self.momentum)
        for iter in tqdm(range(self.num_steps), total=self.num_steps):
            # get a batch of data
            batch_data = self.get_batch_data()
            optimizer.zero_grad()
            # calculate the loss of the model on batched dataset through task-specified calculator
            loss = self.calculator.train_one_step(model, batch_data, self.modalities)['loss']
            loss.backward()
            optimizer.step()
        torch.cuda.empty_cache()
        model.to('cpu')
        
        return model

    def unpack(self, received_pkg):
        """
        Unpack the package received from the server
        :param
            received_pkg: a dict contains the global model as default
        :return:
            the unpacked information that can be rewritten
        """
        # unpack the received package
        return received_pkg['prompt_pools']
        
    def reply(self, svr_pkg):
        v_prompt, t_prompt = self.unpack(svr_pkg)
        self.model = self.train(self.model)
        cpkg = self.pack(self.model.to('cpu'))
        return cpkg
            
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
        