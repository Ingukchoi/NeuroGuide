DEBUG_MODE=False
USE_CUDA = True
CUDA_DEVICE_NUM = 0

##########################################################################################
import os
import logging
os.chdir(os.path.dirname(os.path.abspath(__file__)))
from utils import create_logger, copy_all_src

##########################################################################################
from cluTSPTester import cluTSPTester as Tester
env_params = {
    'n_nodes': 20,
    'n_cluster' :3,
    'multi_start': 1,
    'eval_seed': 1235
}
model_params = {
    'input_dimension' :16,
    'embedding_dim': 128,
    'sqrt_embedding_dim': 128**(1/2),
    'encoder_layer_num': 6,
    'qkv_dim': 16,
    'head_num': 8,
    'logit_clipping': 10,
    'ff_hidden_dim': 512
}
tester_params = {
    'use_cuda': USE_CUDA,
    'model_load': {
        'path': './result/saved_clutsp20_model',  # NeuroGuide_20 directory path of pre-trained model.
        'epoch': 200,  # epoch version of pre-trained model to laod.
    },
    'cuda_device_num': CUDA_DEVICE_NUM,
    'epochs': 1,
    'test_episodes': 1 * 1000,
    'test_batch_size':1000,
    'test_mode' :'greedy', # or 'sampling'
    'test_sampling_size': 1280
}

logger_params = {
    'log_file': {
        'desc': 'test__tsp_n20',
        'filename': 'run_log'
    }
}

##########################################################################################
def main():

    if DEBUG_MODE:
        _set_debug_mode()

    create_logger(**logger_params)
    _print_config()

    tester = Tester(env_params=env_params,
                    model_params=model_params,
                    tester_params=tester_params)

    copy_all_src(tester.result_folder)

    tester.run()


def _set_debug_mode():
    global tester_params
    tester_params['test_episodes'] = 100


def _print_config():
    logger = logging.getLogger('root')
    logger.info('DEBUG_MODE: {}'.format(DEBUG_MODE))
    logger.info('USE_CUDA: {}, CUDA_DEVICE_NUM: {}'.format(USE_CUDA, CUDA_DEVICE_NUM))
    [logger.info(g_key + "{}".format(globals()[g_key])) for g_key in globals().keys() if g_key.endswith('params')]
    
##########################################################################################
if __name__ == "__main__":
    main()