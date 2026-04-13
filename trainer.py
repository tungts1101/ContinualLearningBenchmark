import sys
import logging
import copy
import time
import torch
from utils import factory
from utils.data_manager import DataManager
from utils.toolkit import count_parameters
import os
import numpy as np


def train(args):
    seed_list = copy.deepcopy(args["seed"])
    device = copy.deepcopy(args["device"])

    for seed in seed_list:
        args["seed"] = seed
        args["device"] = device
        _train(args)


def _train(args):

    logs_name = "logs/{}/{}".format(args["model_name"], args["dataset"])

    if not os.path.exists(logs_name):
        os.makedirs(logs_name)

    logfilename = "logs/{}/{}/{}_{}_{}" .format(
        args["model_name"],
        args["dataset"],
        args["prefix"],
        args["seed"],
        args["backbone_type"],
    )
    # Reset handlers so each seed gets its own clean log file
    root = logging.getLogger()
    for h in root.handlers[:]:
        root.removeHandler(h)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(filename)s] => %(message)s",
        handlers=[
            logging.FileHandler(filename=logfilename + ".log"),
            logging.StreamHandler(sys.stdout),
        ],
    )

    _set_random(args["seed"])
    _set_device(args)
    print_args(args)

    data_manager = DataManager(
        args["dataset"],
        args["shuffle"],
        args["seed"],
        args["init_cls"],
        args["increment"],
        args,
    )
    
    args["nb_classes"] = data_manager.nb_classes # update args
    args["nb_tasks"] = data_manager.nb_tasks
    model = factory.get_model(args["model_name"], args)

    cnn_curve, nme_curve = {"top1": [], "top5": []}, {"top1": [], "top5": []}
    cnn_matrix, nme_matrix = [], []

    _t_start          = time.time()
    _total_train_time = 0.0
    _total_eval_time  = 0.0
    _task_params      = []   # per-task param report (before training, matches KCEA pattern)
    _task_storage     = []   # per-task storage report (after after_task)
    _peak_ram         = 0    # peak (model weights + RAM statistics) observed across tasks

    for task in range(data_manager.nb_tasks):
        # Param snapshot BEFORE training — records what is trainable this task
        param_rep = model.get_param_report()
        logging.info("All params: {}".format(param_rep["total"]))
        logging.info("Trainable params: {}".format(param_rep["trainable"]))
        logging.info("  backbone trainable: {}  task-specific: {}".format(
            param_rep["backbone"], param_rep["task_specific"]))
        _task_params.append({"task": task, **param_rep})

        t_train = time.time()
        model.incremental_train(data_manager)
        _total_train_time += time.time() - t_train

        t_eval = time.time()
        cnn_accy, nme_accy = model.eval_task()
        _total_eval_time += time.time() - t_eval

        model.after_task()

        # Storage snapshot AFTER task is committed
        srep = model.get_storage_report()
        weight_bytes = sum(p.numel() * p.element_size() for p in model._network.parameters())
        srep['weight_bytes'] = weight_bytes
        _task_storage.append(srep)
        _peak_ram = max(_peak_ram, weight_bytes + srep['ram_bytes'])

        def _fmt_grouped(d): return {k: float(v) for k, v in d.items()}

        if nme_accy is not None:
            logging.info("CNN: {}".format(_fmt_grouped(cnn_accy["grouped"])))
            logging.info("NME: {}".format(_fmt_grouped(nme_accy["grouped"])))

            cnn_keys = [key for key in cnn_accy["grouped"].keys() if '-' in key]    
            cnn_values = [cnn_accy["grouped"][key] for key in cnn_keys]
            cnn_matrix.append(cnn_values)

            nme_keys = [key for key in nme_accy["grouped"].keys() if '-' in key]
            nme_values = [nme_accy["grouped"][key] for key in nme_keys]
            nme_matrix.append(nme_values)

            cnn_curve["top1"].append(cnn_accy["top1"])
            cnn_curve["top5"].append(cnn_accy["top5"])

            nme_curve["top1"].append(nme_accy["top1"])
            nme_curve["top5"].append(nme_accy["top5"])

            logging.info("CNN top1 curve: {}".format([float(x) for x in cnn_curve["top1"]]))
            logging.info("CNN top5 curve: {}".format([float(x) for x in cnn_curve["top5"]]))
            logging.info("NME top1 curve: {}".format([float(x) for x in nme_curve["top1"]]))
            logging.info("NME top5 curve: {}\n".format([float(x) for x in nme_curve["top5"]]))

            print('Average Accuracy (CNN):', sum(cnn_curve["top1"])/len(cnn_curve["top1"]))
            print('Average Accuracy (NME):', sum(nme_curve["top1"])/len(nme_curve["top1"]))

            logging.info("Average Accuracy (CNN): {}".format(sum(cnn_curve["top1"])/len(cnn_curve["top1"])))
            logging.info("Average Accuracy (NME): {}".format(sum(nme_curve["top1"])/len(nme_curve["top1"])))
        else:
            logging.info("No NME accuracy.")
            logging.info("CNN: {}".format(_fmt_grouped(cnn_accy["grouped"])))

            cnn_keys = [key for key in cnn_accy["grouped"].keys() if '-' in key]
            cnn_values = [cnn_accy["grouped"][key] for key in cnn_keys]
            cnn_matrix.append(cnn_values)

            cnn_curve["top1"].append(cnn_accy["top1"])
            cnn_curve["top5"].append(cnn_accy["top5"])

            logging.info("CNN top1 curve: {}".format([float(x) for x in cnn_curve["top1"]]))
            logging.info("CNN top5 curve: {}\n".format([float(x) for x in cnn_curve["top5"]]))

            print('Average Accuracy (CNN):', sum(cnn_curve["top1"])/len(cnn_curve["top1"]))
            logging.info("Average Accuracy (CNN): {} \n".format(sum(cnn_curve["top1"])/len(cnn_curve["top1"])))

    if len(cnn_matrix) > 0:
        logging.info("Accuracy matrix (CNN):")
        for row in cnn_matrix:
            logging.info("  {}".format([round(float(v), 2) for v in row]))
        if len(cnn_matrix) > 1:
            np_acctable = np.zeros([task + 1, task + 1])
            for idxx, line in enumerate(cnn_matrix):
                np_acctable[idxx, :len(line)] = np.array(line)
            np_acctable = np_acctable.T
            forgetting = float(np.mean((np.max(np_acctable, axis=1) - np_acctable[:, task])[:task]))
            logging.info("Forgetting (CNN): {:.2f}".format(forgetting))

    if len(nme_matrix) > 0:
        logging.info("Accuracy matrix (NME):")
        for row in nme_matrix:
            logging.info("  {}".format([round(float(v), 2) for v in row]))
        if len(nme_matrix) > 1:
            np_acctable = np.zeros([task + 1, task + 1])
            for idxx, line in enumerate(nme_matrix):
                np_acctable[idxx, :len(line)] = np.array(line)
            np_acctable = np_acctable.T
            forgetting = float(np.mean((np.max(np_acctable, axis=1) - np_acctable[:, task])[:task]))
            logging.info("Forgetting (NME): {:.2f}".format(forgetting))

    _log_summary(args, _t_start, _total_train_time, _total_eval_time,
                 model._total_data_time, _task_params, _task_storage, _peak_ram, logfilename)


def _log_summary(args, t_start, total_train_time, total_eval_time,
                 total_data_time, task_params, task_storage, peak_ram, logfilename):
    def _mb(b): return b / 1024 ** 2

    wall_time  = time.time() - t_start
    other_time = max(wall_time - total_train_time - total_eval_time - total_data_time, 0.0)
    active_time = wall_time - total_eval_time - total_data_time

    # ── Trainable params ──────────────────────────────────────────────────────
    peak_trainable     = max((t["trainable"] for t in task_params), default=0)
    peak_total_at_task = next((t["total"] for t in task_params if t["trainable"] == peak_trainable), 0)
    last_trainable     = task_params[-1]["trainable"] if task_params else 0
    last_total         = task_params[-1]["total"] if task_params else 0

    # ── Storage ───────────────────────────────────────────────────────────────
    final_weight_bytes = task_storage[-1]["weight_bytes"] if task_storage else 0
    final_ram_bytes    = task_storage[-1]["ram_bytes"] if task_storage else 0
    final_disk_bytes   = task_storage[-1]["disk_bytes"] if task_storage else 0
    final_disk_files   = task_storage[-1]["n_disk_files"] if task_storage else 0

    logging.info("\n" + "=" * 80)
    logging.info("[Summary] ╔══ END-OF-RUN RESOURCE REPORT ══╗")
    logging.info(f"[Summary]   Method : {args['model_name']}  |  Dataset : {args['dataset']}  |  Seed : {args['seed']}")
    logging.info("[Summary]")

    # Time
    logging.info(f"[Summary] ── Wall-clock time: {wall_time:.1f}s ──────────────────────────")
    logging.info(f"[Summary]   Training (SGD, all tasks)              : {total_train_time:.1f}s  ({total_train_time/wall_time*100:.1f}%)")
    logging.info(f"[Summary]   Evaluation / test inference            : {total_eval_time:.1f}s  ({total_eval_time/wall_time*100:.1f}%)")
    logging.info(f"[Summary]   Data loading (train batches)           : {total_data_time:.1f}s  ({total_data_time/wall_time*100:.1f}%)")
    logging.info(f"[Summary]   Other (I/O, misc)                      : {other_time:.1f}s  ({other_time/wall_time*100:.1f}%)")
    logging.info(f"[Summary]   ──────────────────────────────────────────────────────────────")
    logging.info(f"[Summary]   Active compute (wall − eval − data)    : {active_time:.1f}s  ({active_time/wall_time*100:.1f}%)")
    logging.info("[Summary]")

    # Storage
    logging.info(f"[Summary] ── Storage ─────────────────────────────────────────────────────")
    logging.info(f"[Summary]   Model weights in RAM (final task)      : {_mb(final_weight_bytes):.1f} MB")
    logging.info(f"[Summary]   Peak RAM (weights + statistics)        : {_mb(peak_ram):.1f} MB")
    logging.info(f"[Summary]   On-disk checkpoints                    : {_mb(final_disk_bytes):.1f} MB  ({final_disk_files} files)")
    logging.info(f"[Summary]   RAM statistics (means/covs/etc.)       : {_mb(final_ram_bytes):.1f} MB")
    if task_storage and task_storage[-1].get('detail'):
        for attr, b in task_storage[-1]['detail'].items():
            logging.info(f"[Summary]     {attr}: {_mb(b):.1f} MB")
    logging.info("[Summary]")

    # Parameters
    logging.info(f"[Summary] ── Trainable parameters ─────────────────────────────────────────")
    logging.info(f"[Summary]   Total model params (final task)        : {last_total:,}")
    logging.info(f"[Summary]   Peak trainable (any task)              : {peak_trainable:,}  ({peak_trainable*100/max(peak_total_at_task,1):.2f}% of {peak_total_at_task:,} at that task)")
    logging.info(f"[Summary]   Final task trainable                   : {last_trainable:,}  ({last_trainable*100/max(last_total,1):.2f}% of total at final task)")
    logging.info("[Summary]")
    logging.info("=" * 80)


def _set_device(args):
    device_type = args["device"]
    gpus = []

    for device in device_type:
        if device == -1:
            device = torch.device("cpu")
        else:
            device = torch.device("cuda:{}".format(device))

        gpus.append(device)

    args["device"] = gpus


def _set_random(seed=1):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def print_args(args):
    for key, value in args.items():
        logging.info("{}: {}".format(key, value))