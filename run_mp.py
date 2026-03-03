import argparse
import random
import os
import gzip
import json
from copy import deepcopy
import glob
from pprint import pprint
import time
import threading

import numpy as np
import torch
import torch.multiprocessing as mp
torch.multiprocessing.set_start_method('spawn', force=True)
from multiprocessing import Pool

from habitat import logger
from habitat_baselines.common.baseline_registry import baseline_registry

from vlnce_baselines.config.default import get_config
from vlnce_baselines.common.utils import seed_everything
    
def get_episode_ids_from_config(config):
    data_path = config.TASK_CONFIG.DATASET.DATA_PATH
    split = config.TASK_CONFIG.DATASET.SPLIT
    if _is_rxr_dataset(config): role = config.TASK_CONFIG.DATASET.ROLES
    print(split)
    if _is_rxr_dataset(config): data_path = data_path.format(split=split,role=role[0])
    else: data_path = data_path.format(split=split)
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Episode data file not found: {data_path}")

    # 读取 episodes
    if data_path.endswith('.gz'):
        open_fn = gzip.open
        mode = 'rt'
    else:
        open_fn = open
        mode = 'r'

    with open_fn(data_path, mode) as f:
        data = json.load(f)
        episodes = data.get("episodes", [])
        episode_ids = [e["episode_id"] for e in episodes]
    # print(episode_ids)
    return episode_ids

RXR_FIXED_100 = ['9048', '7902', '1902', '4891', '8038', '1768', '9801', '5368', '5796', '3740', '593', '2169', '6736', '8694', '7420', '648', '8333', '7589', '3749', '1827', '8578', '1326', '10371', '721', '9655', '3177', '9424', '6437', '4280', '5073', '6456', '6066', '6584', '3524', '6341', '9619', '8569', '1664', '6027', '9580', '5913', '7107', '2589', '6662', '8544', '6943', '10825', '8885', '8673', '2257', '1491', '8136', '5562', '8969', '6495', '4366', '30', '771', '5844', '8336', '1644', '2578', '10529', '3781', '4192', '7724', '3651', '10649', '3303', '5467', '7376', '3475', '6569', '3169', '2394', '9374', '6878', '8127', '1296', '1436', '4536', '8970', '5387', '6204', '1064', '5577', '5327', '8945', '4095', '10989', '651', '2693', '9459', '2238', '6093', '2601', '5788', '5527', '5309', '905']

def _is_rxr_dataset(cfg) -> bool:
    try:
        dp = getattr(cfg.TASK_CONFIG.DATASET, 'DATA_PATH', None)
        if dp and 'rxr' in str(dp).lower():
            return True
        name = (
            getattr(cfg.TASK_CONFIG.DATASET, 'DATASET', None)
            or getattr(cfg.TASK_CONFIG.DATASET, 'NAME', None)
            or getattr(cfg.TASK_CONFIG.DATASET, 'DATASET_NAME', None)
        )
        if name and 'rxr' in str(name).lower():
            return True
    except Exception:
        pass
    return False

def run_exp(exp_name: str, exp_config: str, 
            run_type: str, nprocesses: int, opts=None, use_rxr_100: bool = False) -> None:
    r"""Runs experiment given mode and config

    Args:
        exp_config: path to config file.
        run_type: "train" or "eval.
        opts: list of strings of additional config options.

    Returns:
        None.
    """
    config = get_config(exp_config, opts)
    config.defrost()
    config.TENSORBOARD_DIR += exp_name
    config.CHECKPOINT_FOLDER += exp_name
    config.EVAL_CKPT_PATH_DIR += exp_name
    config.RESULTS_DIR += exp_name
    config.VIDEO_DIR += exp_name
    config.LOG_FILE = exp_name + '_' + config.LOG_FILE
    config.freeze()
    
    os.makedirs(config.RESULTS_DIR, exist_ok=True)
    os.makedirs(config.EVAL_CKPT_PATH_DIR, exist_ok=True)
    os.system("mkdir -p data/logs/running_log")
    logger.add_filehandler('data/logs/running_log/' + config.LOG_FILE)
    logger.info(f"hyper parameters:\n{config.EVAL}")
    
    # dataset split, start multi-processes
    num_devices = torch.cuda.device_count()
    print(f'num devices: {num_devices}, num processes: {nprocesses}')

    # episode_ids = list(llm_reply_dataset.keys())
    episode_ids = get_episode_ids_from_config(config)
    total_available = len(episode_ids)
    if use_rxr_100 and _is_rxr_dataset(config):
        fixed_set = set(RXR_FIXED_100)
        available_set = set(map(str, episode_ids))
        selected = [eid for eid in RXR_FIXED_100 if eid in available_set]
        missing = list(fixed_set - available_set)
        if missing:
            print(f"Warning: {len(missing)} of the fixed RXR episode ids are not present in current dataset. They will be skipped.")
        episode_ids = selected
        print(f"Using fixed RXR 100 list: selected {len(episode_ids)} from {total_available} available episodes")
    else:
        if use_rxr_100 and not _is_rxr_dataset(config):
            print("--use-rxr-100 specified but dataset is not RxR; using all episodes.")
        print(f"Using all {total_available} episodes (no sampling)")
    print(f"Total episodes to process: {len(episode_ids)}")
    split_episode_ids = [episode_ids[i::nprocesses] for i in range(nprocesses)]
    
    # 打印每个进程分配的episode数量
    for i, ep_ids in enumerate(split_episode_ids):
        print(f"Process {i}: {len(ep_ids)} episodes")
    
    configs = []
    for i, ep_ids in enumerate(split_episode_ids):    
        if len(ep_ids) == 0:  # 跳过没有分配到episode的进程
            print(f"Warning: Process {i} has no episodes assigned, skipping")
            continue
            
        shared_config = deepcopy(config)
        shared_config.defrost()
        device_num = i % num_devices
        shared_config.local_rank = i
        shared_config.world_size = nprocesses
        shared_config.TORCH_GPU_ID = device_num
        shared_config.TORCH_GPU_IDS = [device_num]
        shared_config.SIMULATOR_GPU_IDS = [device_num]
        shared_config.TASK_CONFIG.DATASET.EPISODES_ALLOWED = ep_ids
        shared_config.freeze()
        configs.append(shared_config)
        print(f"Process {i}: GPU {device_num}, {len(ep_ids)} episodes")
    
    print(f"Actually starting {len(configs)} processes")
    
    pool = Pool(processes=len(configs))
    try:
        print(f"Starting multiprocessing with {len(configs)} workers...")
        start_time = time.time()
        
        result = pool.map_async(worker, configs)
        pool.close()
        
        results = result.get(timeout=7200)
        pool.join()
        
        successful = sum(1 for r in results if r)
        failed = len(results) - successful
        total_time = time.time() - start_time
        
        print(f"Multiprocessing completed in {total_time:.0f}s")
        print(f"Successful workers: {successful}/{len(results)}")
        if failed > 0:
            print(f"Failed workers: {failed}")
            
    except Exception as e:
        print(f"Error occurred: {e}")
        pool.terminate()
        pool.join()
        print("All processes terminated")
        return
    fns = glob.glob(config.CHECKPOINT_FOLDER + '/stats_ep_ckpt_*.json')
    summary = {}
    for fn in fns:
        with open(fn, 'r') as f:
            summary.update(json.load(f))
    summary_metrics = {
        "steps_taken": [],
        "distance_to_goal": [],
        "success": [],
        "oracle_success": [],
        "path_length": [],
        "spl": [],
        "ndtw": [],
        "sdtw": [],
    }
    for epid, metric in summary.items():
        for k, v in metric.items():
            summary_metrics[k].append(v)
    for k, v in summary_metrics.items():
        summary_metrics[k] = np.mean(v)
    pprint(summary_metrics)
    with open(config.CHECKPOINT_FOLDER + '/stats_ckpt_val_unseen.json', 'w') as f:
        json.dump(summary_metrics, f, indent=2)
    
    try:
        from vlnce_baselines.lavira_main import merge_model_usage_stats
        print("\n" + "="*50)
        print("MERGING MODEL USAGE STATISTICS...")
        print("="*50)
        merge_model_usage_stats(config.CHECKPOINT_FOLDER, "val_unseen")
    except Exception as e:
        print(f"Error merging model usage statistics: {e}")

def worker(config):
    try:
        worker_log_file = f"data/logs/running_log/worker_{config.local_rank}_{config.LOG_FILE}"
        logger.add_filehandler(worker_log_file)
        
        print(f"Worker started: local_rank={config.local_rank}, device={config.TORCH_GPU_ID}")
        import sys
        sys.stdout.flush()
        logger.info(f"Worker {config.local_rank} started on GPU {config.TORCH_GPU_ID}")
        logger.info(f"Worker {config.local_rank} processing {len(config.TASK_CONFIG.DATASET.EPISODES_ALLOWED)} episodes")
        
        seed_everything(config.TASK_CONFIG.SEED)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = False
        if torch.cuda.is_available():
            torch.set_num_threads(1)

        TRAINER = baseline_registry.get_trainer(config.TRAINER_NAME)
        assert TRAINER is not None, f"{config.TRAINER_NAME} is not supported"
        print(f"Worker {config.local_rank}: Starting trainer")
        logger.info(f"Worker {config.local_rank}: Starting trainer")
        trainer = TRAINER(config, r2r=not _is_rxr_dataset(config))
        trainer.eval()
        print(f"Worker {config.local_rank}: Completed successfully")
        logger.info(f"Worker {config.local_rank}: Completed successfully")
        return True
    except Exception as e:
        print(f"Worker {config.local_rank} failed with error: {str(e)}")
        logger.error(f"Worker {config.local_rank} failed with error: {str(e)}")
        import traceback
        traceback.print_exc()
        logger.error(traceback.format_exc())
        return False

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--exp_name",
        type=str,
        default="test",
        required=True,
        help="experiment id that matches to exp-id in Notion log",
    )
    parser.add_argument(
        "--run-type",
        choices=["eval"],
        required=True,
        help="run type of the experiment(train, eval, inference), only eval for zero-shot vln",
    )
    parser.add_argument(
        "--nprocesses",
        type=int,
        default=1,
        help="number of processes",
    )
    parser.add_argument(
        "--use-rxr-100",
        action="store_true",
        help="If set and dataset is RxR, use fixed 100-episode list; otherwise no sampling.",
    )
    parser.add_argument(
        "--exp-config",
        type=str,
        required=True,
        help="path to config yaml containing info about experiment",
    )
    parser.add_argument(
        "opts",
        default=None,
        nargs=argparse.REMAINDER,
        help="Modify config options from command line",
    )
    args = parser.parse_args()
    print(args)

    mp.set_start_method('spawn', force=True)
    run_exp(**vars(args))