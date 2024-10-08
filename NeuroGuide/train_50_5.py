DEBUG_MODE = False
USE_CUDA = True
CUDA_DEVICE_NUM = 0

##########################################################################################
import os
from utils import create_logger, copy_all_src

os.chdir(os.path.dirname(os.path.abspath(__file__)))

##########################################################################################
import logging
from cluTSPTrainer import cluTSPTrainer as Trainer

env_params = {
    'n_nodes': 50,
    'n_cluster' :5,
    'multi_start': 50
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

optimizer_params = {
    'lr': 1e-4
}

trainer_params = {
    'use_cuda': USE_CUDA,
    'cuda_device_num': CUDA_DEVICE_NUM,
    'epochs': 200,
    'train_episodes': 200 * 64, # iteration per epcoh
    'train_batch_size': 32
}

logger_params = {
    'log_file': {
        'desc': 'train__tsp_n50',
        'filename': 'run_log'
    }
}

##########################################################################################
def main():
    if DEBUG_MODE:
        _set_debug_mode()

    create_logger(**logger_params)
    _print_config()

    trainer = Trainer(env_params=env_params,
                      model_params=model_params,
                      optimizer_params=optimizer_params,
                      trainer_params=trainer_params)
    
    copy_all_src(trainer.result_folder)
    trainer.run()

def _set_debug_mode():
    global trainer_params
    trainer_params['epochs'] = 2
    trainer_params['train_episodes'] = 10
    trainer_params['train_batch_size'] = 4


def _print_config():
    logger = logging.getLogger('root')
    logger.info('DEBUG_MODE: {}'.format(DEBUG_MODE))
    logger.info('USE_CUDA: {}, CUDA_DEVICE_NUM: {}'.format(USE_CUDA, CUDA_DEVICE_NUM))
    [logger.info(g_key + "{}".format(globals()[g_key])) for g_key in globals().keys() if g_key.endswith('params')]

##########################################################################################

if __name__ == "__main__":
    main()