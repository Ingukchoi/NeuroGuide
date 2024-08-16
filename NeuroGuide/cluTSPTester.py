import torch
import os
from logging import getLogger
from CluTSPModel import CluTSPSolver as Model
from Data_generation import TSPDataset
from utils import *


class cluTSPTester:
    def __init__(self,
                 env_params,
                 model_params,
                 tester_params):

        # save arguments
        self.env_params = env_params
        self.model_params = model_params
        self.tester_params = tester_params

        # result folder, logger
        self.logger = getLogger(name='tester')
        self.result_folder = get_result_folder()

        self.eval_size = self.tester_params['test_batch_size']
        self.problem_size = self.env_params['n_nodes']
        self.cluster_size = self.env_params['n_cluster']
        self.multi_start_size=self.env_params['multi_start']
        self.eval_seed = self.env_params['eval_seed']
        self.test_mode = self.tester_params['test_mode']
        self.sampling_size = self.tester_params['test_sampling_size']

        # cuda
        USE_CUDA = self.tester_params['use_cuda']
        if USE_CUDA:
            cuda_device_num = self.tester_params['cuda_device_num']
            torch.cuda.set_device(cuda_device_num)
            device = torch.device('cuda', cuda_device_num)
            torch.set_default_tensor_type('torch.cuda.FloatTensor')
        else:
            device = torch.device('cpu')
            torch.set_default_tensor_type('torch.FloatTensor')
        self.device = device

        #MODEL
        self.model = Model(**self.model_params)

        # Parameter load
        model_load = tester_params['model_load']
        checkpoint_fullname = '{path}/checkpoint-{epoch}.pt'.format(**model_load)
        checkpoint = torch.load(checkpoint_fullname, map_location=device)
        self.model.load_state_dict(checkpoint)

        #Utility
        self.time_estimator = TimeEstimator()

    def run(self):
        self.time_estimator.reset()

        score_AM = AverageMeter()
        aug_score_AM = AverageMeter()

        test_num_episode = self.tester_params['test_episodes']
        episode = 0

        while episode < test_num_episode:

            remaining = test_num_episode - episode
            batch_size = min(self.tester_params['test_batch_size'], remaining)

            score = self._test_one_batch(batch_size)

            score_AM.update(score, batch_size)

            episode += batch_size

            ############################
            # Logs
            ############################
            elapsed_time_str, remain_time_str = self.time_estimator.get_est_string(episode, test_num_episode)
            self.logger.info("episode {:3d}/{:3d}, Elapsed[{}], Remain[{}], score:{:.4f}".format(
                episode, test_num_episode, elapsed_time_str, remain_time_str, score))

            all_done = (episode == test_num_episode)

            if all_done:
                self.logger.info(" *** Test Done *** ")
                self.logger.info(" Evaluation score: {:.4f} ".format(score_AM.avg))

    def _test_one_batch(self, batch_size):

        #Ready
        ###############################################
        self.model.eval()
        with torch.no_grad():
            if self.test_mode =='greedy':
                self.model.decoder.set_node_select('greedy')
                self.model.decoder.set_cluster_select('greedy')

                eval_dataset = TSPDataset(self.problem_size, batch_size, self.cluster_size, seed=self.eval_seed)

                cost, _, _ = self.model(eval_dataset.data, batch_size, self.problem_size, self.multi_start_size)

                test_score = cost.mean()

            else:
                self.model.decoder.set_node_select('sampling')
                self.model.decoder.set_cluster_select('sampling')

                eval_dataset = TSPDataset(self.problem_size, batch_size, self.cluster_size, seed=self.eval_seed)

                sampling_list=list()

                for step in range(self.sampling_size):
                    cost, _, _ = self.model(eval_dataset.data, batch_size, self.problem_size, self.multi_start_size)

                    sampling_list.append(cost)
                sampling_list = torch.stack(sampling_list)
                best_score = torch.min(sampling_list, dim=0)[0]
                test_score = best_score.mean()

        return test_score.item()