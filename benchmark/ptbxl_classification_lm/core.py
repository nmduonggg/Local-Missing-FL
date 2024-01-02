from .dataset import PTBXLReduceDataset
from benchmark.toolkits import DefaultTaskGen
from benchmark.toolkits import ClassificationCalculator
from benchmark.toolkits import IDXTaskPipe
import os
import ujson
import importlib
import random
import torch
from torch.utils.data import DataLoader, Subset
from sklearn.metrics import accuracy_score
import numpy as np
import warnings
warnings.filterwarnings('ignore')

class ClientSubset(Subset):
    """Custom Subset for local missing setting
    """
    def __init__(self, dataset, indices):
        super(ClientSubset, self).__init__(dataset, indices)
        self.x_missing = dict()
    
    def local_missing_setup(self, leads, ps, pm):
        """
        Local missing setup for client dataset. Will be called after clients have their own datasets
        args:
            leads: client leads
        """
        random.seed(42)
        local_missing_leads = sorted(leads, key=lambda x: random.random())[:int(len(leads)*pm)]
        missing_indices = self.indices[:int(len(self.indices)*ps)]
        print(f"Missing leads {local_missing_leads} from {leads} with {ps*100}%")
        x_missing = {}
        
        for idx in missing_indices: # idx in self.dataset
            x_orig, y = self.dataset[idx]
            x_new = x_orig[:, :]    # 12 x D
            for lead in local_missing_leads:
                x_new[lead, :] = 0
            x_missing[idx] = (x_new, y)
            
        self.x_missing = x_missing
    
    def __getitem__(self, idx):
        if isinstance(idx, list):
            outs = list()
            for i in idx:
                orig_idx = idx[i]
                out = self.x_missing[orig_idx] if orig_idx in self.x_missing else self.dataset[orig_idx]
                outs.append(out)
            return outs
        orig_idx = self.indices[idx]
        out = self.x_missing[orig_idx] if orig_idx in self.x_missing else self.dataset[orig_idx]
        return out
    
    def __getitems__(self, indices):
        # add batched sampling support when parent dataset supports it.
        # see torch.utils.data._utils.fetch._MapDatasetFetcher
        outs = list()
        for idx in indices:
            orig_idx = self.indices[idx]
            out = self.x_missing[orig_idx] if orig_idx in self.x_missing else self.dataset[orig_idx]
            outs.append(out)
        return outs
    
    def __len__(self):
        return len(self.indices)

class TaskPipe(IDXTaskPipe):
    TaskDataset = ClientSubset
    @classmethod
    def load_task(cls, task_path):
        with open(os.path.join(task_path, 'data.json'), 'r') as inf:
            feddata = ujson.load(inf)
        class_path = feddata['datasrc']['class_path']
        class_name = feddata['datasrc']['class_name']
        
        origin_class = getattr(importlib.import_module(class_path), class_name)
        origin_train_data = cls.args_to_dataset(origin_class, feddata['datasrc']['train_args'])
        origin_test_data = cls.args_to_dataset(origin_class, feddata['datasrc']['test_args'])
        test_data = cls.TaskDataset(origin_test_data, [_ for _ in range(len(origin_test_data))])
        train_datas = []
        valid_datas = []
        modalities_list = []
        for name in feddata['client_names']:
            train_data = feddata[name]['dtrain']    # sample idx
            valid_data = feddata[name]['dvalid']
            if cls._cross_validation:
                k = len(train_data)
                train_data.extend(valid_data)
                random.shuffle(train_data)
                all_data = train_data
                train_data = all_data[:k]
                valid_data = all_data[k:]
            if cls._train_on_all:
                train_data.extend(valid_data)
            train_datas.append(cls.TaskDataset(origin_train_data, train_data))
            valid_datas.append(cls.TaskDataset(origin_train_data, valid_data))
            modalities_list.append(feddata[name]['modalities'])
            # modalities_list.append(list(range(12)))
        return train_datas, valid_datas, test_data, feddata['client_names'], modalities_list

def save_task(generator):
    """
    Store the splited indices of the local data in the original dataset (source dataset) into the disk as .json file
    The input 'generator' must have attributes:
        :taskpath: string. the path of storing
        :train_data: the training dataset which is a dict {'x':..., 'y':...}
        :test_data: the testing dataset which is a dict {'x':..., 'y':...}
        :train_cidxs: a list of lists of integer. The splited indices in train_data of the training part of each local dataset
        :valid_cidxs: a list of lists of integer. The splited indices in train_data of the valiadtion part of each local dataset
        :client_names: a list of strings. The names of all the clients, which is used to index the clients' data in .json file
        :source_dict: a dict that contains parameters which is necessary to dynamically importing the original Dataset class and generating instances
                For example, for MNIST using this task pipe, the source_dict should be like:
                {'class_path': 'torchvision.datasets',
                    'class_name': 'MNIST',
                    'train_args': {'root': '"'+MNIST_rawdata_path+'"', 'download': 'True', 'transform': 'transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])','train': 'True'},
                    'test_args': {'root': '"'+MNIST_rawdata_path+'"', 'download': 'True', 'transform': 'transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])', 'train': 'False'}
                }
        :return:
    """
    feddata = {
        'store': 'IDX',
        'client_names': generator.cnames,
        'dtest': [i for i in range(len(generator.test_data))],
        'datasrc': generator.source_dict
    }
    for cid in range(len(generator.cnames)):
        if generator.specific_training_leads:
            # import pdb; pdb.set_trace()
            feddata[generator.cnames[cid]] = {
                'modalities': generator.specific_training_leads[cid],
                'dtrain': generator.train_cidxs[cid],
                'dvalid': generator.valid_cidxs[cid]
            }
        else:
            feddata[generator.cnames[cid]] = {
                'dtrain': generator.train_cidxs[cid],
                'dvalid': generator.valid_cidxs[cid]
            }
    with open(os.path.join(generator.taskpath, 'data.json'), 'w') as outf:
        ujson.dump(feddata, outf)
    return
    

def iid_partition(generator):
    print(generator)
    # import pdb; pdb.set_trace()
    labels = np.unique(generator.train_data.y)
    local_datas = [[] for _ in range(generator.num_clients)]
    for label in labels:
        permutation = np.random.permutation(np.where(generator.train_data.y == label)[0])
        split = np.array_split(permutation, generator.num_clients)
        for i, idxs in enumerate(split):
            local_datas[i] += idxs.tolist()
    # import pdb; pdb.set_trace()
    return local_datas

# def local_holdout(self, local_datas, shuffle=False):
#         """split each local dataset into train data and valid data according the rate."""
#         train_cidxs = []
#         valid_cidxs = []
#         for local_data in local_datas:
#             if shuffle:
#                 np.random.shuffle(local_data)
#             k = int(len(local_data) * (1-self.local_holdout_rate))
#             train_cidxs.append(local_data[:k])
#             valid_cidxs.append(local_data[k:])
#         return train_cidxs, valid_cidxs
    

class TaskGen(DefaultTaskGen):
    def __init__(self, dist_id, num_clients=1, skewness=0.5, local_hld_rate=0.0, seed=0, 
                 percentages=None, missing=False, modal_equality=False, modal_missing_case3=False, 
                 modal_missing_case4=False):
        super(TaskGen, self).__init__(benchmark='ptbxl_classification_lm',
                                      dist_id=dist_id,
                                      num_clients=num_clients,
                                      skewness=skewness,
                                      rawdata_path='./benchmark/RAW_DATA/PTBXL_REDUCE',
                                      local_hld_rate=local_hld_rate,
                                      seed = seed
                                      )
        if self.dist_id == 0:
            self.partition = iid_partition
        # self.local_holdout = local_holdout
        self.num_classes = 10
        self.save_task = save_task
        self.visualize = self.visualize_by_class
        self.source_dict = {
            'class_path': 'benchmark.ptbxl_classification_lm.dataset',
            'class_name': 'PTBXLReduceDataset',
            'train_args': {
                'root': '"'+self.rawdata_path+'"',
                'download': 'True',
                'standard_scaler': 'True',
                'train':'True',
            },
            'test_args': {
                'root': '"'+self.rawdata_path+'"',
                'download': 'True',
                'standard_scaler': 'True',
                'train': 'False'
            }
        }
        
        self.missing = missing
        self.modal_equality = modal_equality
        self.modal_missing_case3 = modal_missing_case3
        self.modal_missing_case4 = modal_missing_case4
        self.specific_training_leads = None
        self.local_holdout_rate = 0.1
        if self.missing and self.num_clients == 20:
            if self.modal_equality:
                self.specific_training_leads = [
                    (4, 7, 8, 9, 10, 11),
                    (0, 2, 5, 7, 9, 11),
                    (1, 2, 3, 7, 9, 11),
                    (1, 3, 4, 6, 7, 9),
                    (0, 1, 4, 5, 10, 11),
                    (0, 1, 2, 3, 8, 9),
                    (0, 1, 3, 6, 7, 8),
                    (2, 3, 4, 5, 7, 11),
                    (0, 3, 4, 7, 10, 11),
                    (1, 3, 4, 5, 7, 10),
                    (0, 3, 4, 9, 10, 11),
                    (0, 2, 3, 4, 7, 8),
                    (1, 3, 5, 6, 7, 8),
                    (0, 1, 5, 7, 8, 10),
                    (0, 6, 7, 8, 9, 11),
                    (0, 4, 5, 6, 7, 8),
                    (0, 5, 6, 7, 8, 9),
                    (0, 1, 2, 3, 5, 9),
                    (3, 4, 5, 7, 8, 9),
                    (1, 5, 7, 8, 9, 11)
                ]
                self.taskname = self.taskname + '_missing_modal_equality'
                self.taskpath = os.path.join(self.task_rootpath, self.taskname)
            elif self.modal_missing_case3:
                self.specific_training_leads = [
                    (1, 11), 
                    (0, 4, 6, 7, 10), 
                    (3, 6), 
                    (1, 2, 3, 4), 
                    (5,), 
                    (0, 1, 3, 5, 6, 9, 10), 
                    (1, 2, 4, 6, 8), 
                    (0, 1, 6), (1, 3, 8), 
                    (1, 3, 9, 10), 
                    (0,), 
                    (0, 3, 7, 8), 
                    (1, 5, 9, 11), 
                    (0, 2, 4, 5, 6, 7, 8), 
                    (0, 1, 2, 5, 6, 8), 
                    (4,), 
                    (1, 5), 
                    (3, 4, 8, 9), 
                    (6, 9), 
                    (0, 1, 3, 7, 8, 9, 10, 11)
                ]
                self.taskname = self.taskname + '_missing_modal_case3'
                self.taskpath = os.path.join(self.task_rootpath, self.taskname)
            elif self.modal_missing_case4:
                self.specific_training_leads = [
                    (1, 2, 3, 8, 10, 11), 
                    (1, 2, 5, 6, 7, 8), 
                    (0, 3, 4, 5, 6, 7, 8, 9, 10), 
                    (2, 5, 7, 9, 10, 11), 
                    (0, 1, 3, 4, 5, 6, 7, 8, 10, 11), 
                    (1, 3, 7, 9, 10, 11), 
                    (0, 1, 2, 3, 4, 6, 8, 9, 10, 11), 
                    (1, 2, 3, 5, 6, 7, 8, 10, 11), 
                    (0, 1, 2, 3, 4, 5, 6, 7, 9, 10), 
                    (2, 3, 4, 5, 7, 8, 11), 
                    (0, 2, 3, 4, 7, 8, 9, 10, 11), 
                    (0, 1, 2, 3, 5, 7, 9, 10, 11), 
                    (1, 3, 4, 5, 9, 10, 11), 
                    (0, 2, 4, 5, 6, 7, 9), 
                    (0, 1, 2, 4, 5, 7, 8, 10, 11), 
                    (1, 2, 4, 5, 6, 9), 
                    (0, 1, 4, 7, 9, 10), 
                    (0, 1, 3, 5, 6, 8, 9, 10, 11), 
                    (0, 3, 5, 6, 9, 10), 
                    (1, 2, 3, 5, 6, 7, 9)
                ]
                self.taskname = self.taskname + '_missing_modal_case4_mifl_local_missing'
                self.taskpath = os.path.join(self.task_rootpath, self.taskname)
            else:
                self.specific_training_leads = [       # 2-6 modalities
                    (4, 5, 8),
                    (4, 5),
                    (2, 3, 5, 9),
                    (1, 3, 7, 8, 11),
                    (5, 6, 8, 9),
                    (0, 2, 3, 5, 8, 9),
                    (0, 2, 3, 5),
                    (0, 1, 3, 5),
                    (0, 3, 5, 10, 11),
                    (1, 4, 6),
                    (8, 9, 11),
                    (0, 3, 5, 6, 7, 11),
                    (2, 3, 4, 5, 7),
                    (0, 4, 7, 8),
                    (0, 3, 4, 6, 7),
                    (1, 5, 6, 7, 8),
                    (0, 1, 3, 4, 10),
                    (2, 4, 5, 7, 9, 11),
                    (3, 4, 5, 8, 10, 11),
                    (0, 1, 3, 7, 9, 11)
                ]
                self.taskname = self.taskname + '_missing_mifl_local_missing'
                self.taskpath = os.path.join(self.task_rootpath, self.taskname)

    # def local_holdout(self, local_datas, shuffle=False):
    #     """split each local dataset into train data and valid data according the rate."""
    #     train_cidxs = []
    #     valid_cidxs = []
    #     for local_data in local_datas:
    #         if shuffle:
    #             np.random.shuffle(local_data)
    #         k = int(len(local_data) * (1-self.local_holdout_rate))
    #         train_cidxs.append(local_data[:k])
    #         valid_cidxs.append(local_data[k:])
    #     return train_cidxs, valid_cidxs

    def load_data(self):
        self.train_data = PTBXLReduceDataset(
            root=self.rawdata_path,
            download=True,
            standard_scaler=True,
            train=True, 
        )
        # self.valid_data = PTBXLReduceDataset(
        #     root=self.rawdata_path,
        #     download=True,
        #     standard_scaler=True,
        #     train=True,
        #     valid=True
        # )
        # import pdb; pdb.set_trace()
        self.test_data = PTBXLReduceDataset(
            root=self.rawdata_path,
            download=True,
            standard_scaler=True,
            train=False,
        )
    
    def local_holdout(self, local_datas, shuffle=False):
        """split each local dataset into train data and valid data according the rate."""
        train_cidxs = []
        valid_cidxs = []
        for local_data in local_datas:
            if shuffle:
                np.random.shuffle(local_data)
            k = int(len(local_data) * (1-self.local_holdout_rate))
            train_cidxs.append(local_data[:k])
            valid_cidxs.append(local_data[k:])
        
        return train_cidxs, valid_cidxs
    
class TaskCalculator(ClassificationCalculator):
    def __init__(self, device, optimizer_name='sgd'):
        super(TaskCalculator, self).__init__(device, optimizer_name)
        self.n_leads = 12
        self.DataLoader = DataLoader

    def train_one_step(self, model, data, leads):
        """
        :param model: the model to train
        :param data: the training dataset
        :return: dict of train-one-step's result, which should at least contains the key 'loss'
        """
        tdata = self.data_to_device(data)
        # import pdb; pdb.set_trace()
        loss = model(tdata[0], tdata[-1], leads)
        return {'loss': loss}

    @torch.no_grad()
    def test(self, model, dataset, leads, batch_size=64, num_workers=0):
        """
        Metric = [mean_accuracy, mean_loss]
        :param model:
        :param dataset:
        :param batch_size:
        :return: [mean_accuracy, mean_loss]
        """
        model.eval()
        if batch_size==-1:batch_size=len(dataset)
        data_loader = self.get_data_loader(dataset, batch_size=batch_size, num_workers=num_workers)
        total_loss = 0.0
        labels = list()
        predicts = list()
        for batch_id, batch_data in enumerate(data_loader):
            batch_data = self.data_to_device(batch_data)
            labels.extend(batch_data[1].cpu().tolist())
            # import pdb; pdb.set_trace()
            
            loss_leads, loss, outputs = model(batch_data[0], batch_data[-1], leads)
            total_loss += loss.item() * len(batch_data[-1])
            predicts.extend(torch.argmax(torch.softmax(outputs, dim=1), dim=1).cpu().tolist())
        labels = np.array(labels)
        predicts = np.array(predicts)
        accuracy = accuracy_score(labels, predicts)
        return {
            'loss': total_loss / len(dataset),
            'acc': accuracy
        }
        
    @torch.no_grad()
    def evaluate(self, model, dataset, leads, batch_size=64, num_workers=0):
        """
        Metric = [mean_accuracy, mean_loss]
        :param model:
        :param dataset:
        :param batch_size:
        :return: [mean_accuracy, mean_loss]
        """
        model.eval()
        if batch_size==-1:batch_size=len(dataset)
        data_loader = self.get_data_loader(dataset, batch_size=batch_size, num_workers=num_workers)
        # import pdb; pdb.set_trace()    

        # total_loss = []
        labels = list()
        predicts = list()
        for batch_id, batch_data in enumerate(data_loader):
            batch_data = self.data_to_device(batch_data)
            labels.extend(batch_data[1].cpu().tolist())
            loss, loss_, outputs = model(batch_data[0], batch_data[-1], leads)
            # import pdb; pdb.set_trace()    
            loss = torch.tensor(loss).to(loss[0].device)
            
            if batch_id == 0:
                total_loss = loss
            else:    
                total_loss = loss + total_loss
            
        
        loss_eval = [loss / (batch_id + 1) for loss in total_loss]
        # loss_eval = [loss for loss in total_loss]
        
        #     total_loss += loss.item() * len(batch_data[-1])
        #     predicts.extend(torch.argmax(torch.softmax(outputs, dim=1), dim=1).cpu().tolist())
        # labels = np.array(labels)
        # predicts = np.array(predicts)
        # accuracy = accuracy_score(labels, predicts)
        # return {
        #     'loss': total_loss / len(dataset),
        #     'acc': accuracy
        # }
        return loss_eval


    @torch.no_grad()
    def server_test(self, model, dataset, leads, batch_size=64, num_workers=0):
        """
        Metric = [mean_accuracy, mean_loss]
        :param model:
        :param dataset:
        :param batch_size:
        :return: [mean_accuracy, mean_loss]
        """
        model.eval()
        if batch_size==-1:batch_size=len(dataset)
        data_loader = self.get_data_loader(dataset, batch_size=batch_size, num_workers=num_workers)
        result = dict() 
        for test_combi_index in range(len(leads)):
            total_loss = 0.0
            labels = list()
            predicts = list()   
            # loss_each_modal = [[] for i in range(self.n_leads)]
            # loss_each_modal = [0]*12
            for batch_id, batch_data in enumerate(data_loader):
                batch_data = self.data_to_device(batch_data)
                labels.extend(batch_data[1].cpu().tolist())
                loss_leads, loss, outputs = model(batch_data[0], batch_data[-1], leads[test_combi_index])
                # for i in range(self.n_leads):
                #     loss_each_modal[i] += loss_leads[i] * len(batch_data[-1])
                total_loss += loss.item() * len(batch_data[-1])
                predicts.extend(torch.argmax(torch.softmax(outputs, dim=1), dim=1).cpu().tolist())
            # import pdb; pdb.set_trace()
            # import pdb; pdb.set_trace()
            labels = np.array(labels)
            predicts = np.array(predicts)
            accuracy = accuracy_score(labels, predicts)
            # for i in range(self.n_leads):
            #     result['loss_modal_combi'+str(test_combi_index+1)+'_modal'+str(i+1)] = loss_each_modal[i] / len(dataset)
            result['loss'+str(test_combi_index+1)] = total_loss / len(dataset)
            result['acc'+str(test_combi_index+1)] = accuracy
        # return {
        #     'loss': total_loss / len(dataset),
        #     'acc': accuracy
        # }
        # import pdb;pdb.set_trace()
        return result


    @torch.no_grad()
    def full_modal_server_test(self, model, dataset, leads, batch_size=64, num_workers=0):
        """
        Metric = [mean_accuracy, mean_loss]
        :param model:
        :param dataset:
        :param batch_size:
        :return: [mean_accuracy, mean_loss]
        """
        model.eval()
        if batch_size==-1:batch_size=len(dataset)
        data_loader = self.get_data_loader(dataset, batch_size=batch_size, num_workers=num_workers)
        total_loss = 0.0
        labels = list()
        predicts = list()
        for batch_id, batch_data in enumerate(data_loader):
            batch_data = self.data_to_device(batch_data)
            labels.extend(batch_data[1].cpu().tolist())
            loss, outputs = model(batch_data[0], batch_data[-1], leads)
            total_loss += loss.item() * len(batch_data[-1])
            predicts.extend(torch.argmax(torch.softmax(outputs, dim=1), dim=1).cpu().tolist())
        labels = np.array(labels)
        predicts = np.array(predicts)
        accuracy = accuracy_score(labels, predicts)
        return {
            'loss': total_loss / len(dataset),
            'acc': accuracy
        }

    @torch.no_grad()
    def independent_test(self, model, dataset, leads, batch_size=64, num_workers=0):
        """
        Metric = [mean_accuracy, mean_loss]
        :param model:
        :param dataset:
        :param batch_size:
        :return: [mean_accuracy, mean_loss]
        """
        model.eval()
        if batch_size==-1:batch_size=len(dataset)
        data_loader = self.get_data_loader(dataset, batch_size=batch_size, num_workers=num_workers)
        result = dict() 
        for test_combi_index in range(len(leads)):
            labels = list()
            predicts = list()
            for batch_id, batch_data in enumerate(data_loader):
                batch_data = self.data_to_device(batch_data)
                labels.extend(batch_data[1].cpu().tolist())
                predict = model.predict(batch_data[0], batch_data[-1], leads[test_combi_index])
                predicts.extend(predict.argmax(dim=1).cpu().tolist())
            predicts = np.array(predicts)
            accuracy = accuracy_score(labels, predicts)
            result['acc'+str(test_combi_index+1)] = accuracy
        return result
        
    @torch.no_grad()
    def independent_test_detail(self, model, dataset, leads, batch_size=64, num_workers=0):
        model.eval()
        if batch_size==-1:batch_size=len(dataset)
        data_loader = self.get_data_loader(dataset, batch_size=1, num_workers=num_workers)
        labels = list()
        
        fin_output = []
        for batch_id, batch_data in enumerate(data_loader):
            batch_data = self.data_to_device(batch_data)
            labels.extend(batch_data[1].cpu().tolist())
            fin_output.append(model.predict_detail(batch_data[0], batch_data[-1], leads))
        
        return fin_output
    
