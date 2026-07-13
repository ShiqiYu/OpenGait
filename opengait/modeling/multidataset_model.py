import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from .base_model import BaseModel
from .modules import PackSequenceWrapper, HorizontalPoolingPyramid, SetBlockWrapper, ParallelBN1d, SeparateFCs

from utils import np2var, list2var, get_valid_args, ddp_all_gather
from data.transform import get_transform
from einops import rearrange

from torch.cuda.amp import GradScaler, autocast
from utils import get_msg_mgr
import os.path as osp
from .loss_aggregator import LossAggregator
from evaluation import evaluator as eval_functions
from data.dataset import DataSet
import data.sampler as Samplers
from utils import get_valid_args, is_list, is_dict, np2var, ts2np, list2var, get_attr_from
import torch.utils.data as tordata
from data.collate_fn import CollateFn

# Modified from https://github.com/PatrickHua/SimSiam/blob/main/models/simsiam.py
class MultiDatasets(BaseModel):

    def __init__(self, cfgs, training):
        """Initialize the base model.

        Complete the model initialization, including the data loader, the network, the optimizer, the scheduler, the loss.

        Args:
        cfgs:
            All of the configs.
        training:
            Whether the model is in training mode.
        """
        super(BaseModel, self).__init__()
        self.msg_mgr = get_msg_mgr()
        self.cfgs = cfgs
        self.iteration = 0
        self.engine_cfg = cfgs['trainer_cfg'] if training else cfgs['evaluator_cfg']
        if self.engine_cfg is None:
            raise Exception("Initialize a model without -Engine-Cfgs-")

        if training and self.engine_cfg['enable_float16']:
            self.Scaler = GradScaler()

        if isinstance(cfgs['data_cfg'], list):
            dataset_names = "_".join([d['dataset_name'] for d in cfgs['data_cfg']])
            self.save_path = osp.join('output/', dataset_names,cfgs['model_cfg']['model'], self.engine_cfg['save_name'])
        else:
            self.save_path = osp.join('output/', cfgs['data_cfg']['dataset_name'],cfgs['model_cfg']['model'], self.engine_cfg['save_name'])

        self.build_network(cfgs['model_cfg'])
        self.init_parameters()
        self.trainer_trfs = get_transform(cfgs['trainer_cfg']['transform'])

        self.msg_mgr.log_info(cfgs['data_cfg'])
        if training:
            self.train_loader = self.get_loader(cfgs['data_cfg'], train=True)
        if not training or self.engine_cfg['with_test']:
            self.test_loader = self.get_loader(cfgs['data_cfg'][0], train=False)
            self.evaluator_trfs = get_transform(cfgs['evaluator_cfg']['transform'])

        self.device = torch.distributed.get_rank()
        torch.cuda.set_device(self.device)
        self.to(device=torch.device(
            "cuda", self.device))

        if training:
            self.loss_aggregator = LossAggregator(cfgs['loss_cfg'])
            self.optimizer = self.get_optimizer(self.cfgs['optimizer_cfg'])
            self.scheduler = self.get_scheduler(cfgs['scheduler_cfg'])
        self.train(training)
        restore_hint = self.engine_cfg['restore_hint']
        if restore_hint != 0:
            self.resume_ckpt(restore_hint)

    @ staticmethod
    def run_train(model):
        """Accept the instance object(model) here, and then run the train loop."""
        iters_list = {}

        dataset_lengths = [len(l.dataset)/10 if 'gaitlu' in l.dataset.dataset_name.lower() else len(l.dataset) for l in model.train_loader]
        dataset_lengths = torch.tensor(dataset_lengths, dtype=torch.float32)
        probs = torch.softmax(torch.log(dataset_lengths) / 3.0, dim=0)
        g = torch.Generator()

        for _ in range(model.engine_cfg['total_iter']):
            # i = model.iteration % len(model.train_loader)
            g.manual_seed(model.iteration)
            i = torch.multinomial(probs, num_samples=1, generator=g).item()
            if i not in iters_list:
                iters_list[i] = iter(model.train_loader[i])
            inputs = next(iters_list[i])

        # for inputs in model.train_loader:
            ipts = model.inputs_pretreament(inputs)
            with autocast(enabled=model.engine_cfg['enable_float16']):
                retval = model(ipts)
                training_feat, visual_summary = retval['training_feat'], retval['visual_summary']
                del retval
            loss_sum, loss_info = model.loss_aggregator(training_feat)

            ok = model.train_step(loss_sum)
            if not ok:
                continue

            visual_summary.update(loss_info)
            visual_summary['scalar/learning_rate'] = model.optimizer.param_groups[0]['lr']
            loss_info['scalar/opt/lr'] = model.optimizer.param_groups[0]['lr']

            model.msg_mgr.train_step(loss_info, visual_summary)
            if model.iteration % model.engine_cfg['save_iter'] == 0:
                # save the checkpoint
                model.save_ckpt(model.iteration)

                # run test if with_test = true
                if model.engine_cfg['with_test']:
                    model.msg_mgr.log_info("Running test...")
                    model.eval()
                    result_dict = MultiDatasets.run_test(model)
                    model.train()
                    if model.cfgs['trainer_cfg']['fix_BN']:
                        model.fix_BN()
                    if result_dict:
                        model.msg_mgr.write_to_tensorboard(result_dict)
                    model.msg_mgr.reset_time()
            if model.iteration >= model.engine_cfg['total_iter']:
                break

    @ staticmethod
    def run_test(model):
        """Accept the instance object(model) here, and then run the test loop."""
        evaluator_cfg = model.cfgs['evaluator_cfg']
        if torch.distributed.get_world_size() != evaluator_cfg['sampler']['batch_size']:
            raise ValueError("The batch size ({}) must be equal to the number of GPUs ({}) in testing mode!".format(
                evaluator_cfg['sampler']['batch_size'], torch.distributed.get_world_size()))
        rank = torch.distributed.get_rank()
        with torch.no_grad():
            info_dict = model.inference(rank)
        if rank == 0:
            loader = model.test_loader
            label_list = loader.dataset.label_list
            types_list = loader.dataset.types_list
            views_list = loader.dataset.views_list

            info_dict.update({
                'labels': label_list, 'types': types_list, 'views': views_list})

            if 'eval_func' in evaluator_cfg.keys():
                eval_func = evaluator_cfg["eval_func"]
            else:
                eval_func = 'identification'
            eval_func = getattr(eval_functions, eval_func)
            valid_args = get_valid_args(
                eval_func, evaluator_cfg, ['metric'])
            try:
                dataset_name = model.cfgs['data_cfg'][0]['test_dataset_name']
            except:
                dataset_name = model.cfgs['data_cfg'][0]['dataset_name']
            return eval_func(info_dict, dataset_name, **valid_args)

    def get_loader(self, data_cfg, train=True):
        sampler_cfg = self.cfgs['trainer_cfg']['sampler'] if train else self.cfgs['evaluator_cfg']['sampler']
        dataset = []
        dataset_label_set = set()
        if isinstance(data_cfg, list):
            for _ in data_cfg:
                dataset.append(DataSet(_, train))
                dataset_label_set.update(set(dataset[-1].label_set))
        else:
            dataset.append(DataSet(data_cfg, train))
            dataset_label_set.update(set(dataset[-1].label_set))
        dataset_label_set = sorted(list(dataset_label_set))

        Sampler = get_attr_from([Samplers], sampler_cfg['type'])
        vaild_args = get_valid_args(Sampler, sampler_cfg, free_keys=['sample_type', 'type'])

        loader = []
        for _ in dataset:
            sampler = Sampler(_, **vaild_args)
            loader.append(tordata.DataLoader(
                dataset=_,
                batch_sampler=sampler,
                collate_fn=CollateFn(dataset_label_set, sampler_cfg),
                num_workers=1,
                pin_memory=True))
        
        if isinstance(data_cfg, list):
            return loader
        return loader[0]
