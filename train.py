import os
import sys
import argparse
import shutil

import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from configs import config
from data_processes import data_loader
from utilities import general_utilities as util
from data_processes.data_processing import data_preprocessing, load_fold
import container


def run_test(use_gpu, score_containers, inference_type="Test"):
    """ Runs a pre-trained model on the validation folds or test fold

    Args:
        use_gpu: Bool - set True to use GPU for training
        score_containers: tuple - contains dictionaries of the result holders
            (ua, wa, and loss)
        inference_type: str - set to "Validation" for validation folds or
            "Test" for test fold

    Returns: None

    """
    print(f"Running Evaluation on Experiment: {config.MODEL_NAME}\n")

    total_test_ua, total_test_wa, total_test_loss = score_containers

    timer_size = len(seeds) * (config.NUM_FOLDS - 1)
    timer = tqdm(total=timer_size)
    for seed in range(len(seeds)):
        if inference_type == "Validation":
            test_container_wa = container.ResultsContainer(
                config=config,
                data_split=inference_type,
                use_gpu=use_gpu,
                use_weights=False,
                class_weights=None)
            test_container_ua = container.ResultsContainer(
                config=config,
                data_split=inference_type,
                use_gpu=use_gpu,
                use_weights=False,
                class_weights=None)
            tc = [test_container_wa, test_container_ua]
        for exp_fold in range(config.NUM_FOLDS - 1):
            if inference_type != "Validation":
                test_container_wa = container.ResultsContainer(
                    config=config,
                    data_split=inference_type,
                    use_gpu=use_gpu,
                    use_weights=False,
                    class_weights=None)
                test_container_ua = container.ResultsContainer(
                    config=config,
                    data_split=inference_type,
                    use_gpu=use_gpu,
                    use_weights=False,
                    class_weights=None)
                tc = [test_container_wa, test_container_ua]
                fold_to_load = config.NUM_FOLDS - 1
            else:
                fold_to_load = exp_fold
            dataset_dict = load_fold(config=config,
                                     fold=fold_to_load)
            data = dataset_dict["valid_data"]

            MODEL_PATH_WA = f"fold_{exp_fold}_seed_{seed}" \
                            f"_{config.FEATURES_TO_USE}_wa.pth"
            MODEL_PATH_WA = os.path.join(config.EXP_DIR, f"seed_{seed}",
                                         MODEL_PATH_WA)
            MODEL_PATH_UA = f"fold_{exp_fold}_seed_{seed}" \
                            f"_{config.FEATURES_TO_USE}_ua.pth"
            MODEL_PATH_UA = os.path.join(config.EXP_DIR, f"seed_{seed}",
                                         MODEL_PATH_UA)

            test_model = util.setup_model(config=config)
            for p, m in enumerate([MODEL_PATH_WA, MODEL_PATH_UA]):
                test_model = util.setup_model(config=config)
                test_model.load_state_dict(torch.load(m))

                if use_gpu:
                    test_model = test_model.cuda()
                test_model.eval()

                util.forward_pass_test_data(model=test_model,
                                            validation_data=data,
                                            test_container=tc[p],
                                            use_gpu=use_gpu)
            timer.update(1)
            if inference_type != "Validation":
                for p, m in enumerate([MODEL_PATH_WA, MODEL_PATH_UA]):
                    tc[p].__end_epoch__(save_model=False,
                                        model_to_save=(test_model,
                                                       MODEL_PATH_WA))

                    temp_scores = tc[p].__getter__()
                    if p == 0:
                        best_scores = temp_scores
                    else:
                        best_scores["max_ua"] = temp_scores["max_ua"]

                total_test_ua, total_test_wa, total_test_loss = \
                    util.append_scores(config=config,
                                       total_scores_ua=total_test_ua,
                                       total_scores_wa=total_test_wa,
                                       total_scores_loss=total_test_loss,
                                       best_scores=best_scores)

        if inference_type == "Validation":
            for p, m in enumerate([MODEL_PATH_WA, MODEL_PATH_UA]):
                tc[p].__end_epoch__(save_model=False,
                                    model_to_save=(test_model, MODEL_PATH_WA))

                temp_scores = tc[p].__getter__()
                if p == 0:
                    best_scores = temp_scores
                else:
                    best_scores["max_ua"] = temp_scores["max_ua"]

            total_test_ua, total_test_wa, total_test_loss = \
                util.append_scores(config=config,
                                   total_scores_ua=total_test_ua,
                                   total_scores_wa=total_test_wa,
                                   total_scores_loss=total_test_loss,
                                   best_scores=best_scores)
    timer.close()
    util.print_res(config=config,
                   ua=total_test_ua,
                   wa=total_test_wa,
                   loss=total_test_loss,
                   data_type=inference_type)


def run_eval(model_train, validation_data, epoch, model_path, test_container,
             save_model=True):
    """
    Run a model over a validation fold and update the results in the test
    container

    Args:
        model_train: nn.Module - the model to test
        validation_data: dict - holds the validation data and their
            respective labels
        epoch: int - current epoch of training
        model_path: str - location to save the model
        test_container: - container to hold results
        save_model: Bool - set True to save the model

    Returns:
        best_scores: dict - holds the results from the best epoch/model
    """

    util.forward_pass_test_data(model=model_train,
                                validation_data=validation_data,
                                test_container=test_container,
                                use_gpu=gpu)
    test_container.__end_epoch__(save_model=save_model,
                                 model_to_save=(model_train, model_path))
    test_container.__printer__(epoch=epoch)
    test_container.__logger__(logger=logger,
                              epoch=epoch)
    best_scores = test_container.__getter__()
    test_container.__resetter__()

    return best_scores


def train_model(dataset_dict, logger):
    """ Function to setup and train a model

    Args:
        dataset_dict: dict - holds the training data, training labels and
            validation data (dict holding data and labels)
        logger: logger - to record information during training

    Returns:
        best_scores: dict - holds the results from the best epoch/model
    """
    train_data = dataset_dict["train_data"]
    train_targets = dataset_dict["train_targets"]
    validation_data = dataset_dict["valid_data"]

    class_weights = util.get_class_weights(
        labels=train_targets,
        use_class_weights=config.CLASS_WEIGHTS,
        emotions=config.CLASS_DICT_IDX,
        use_cuda=gpu,
        moe=config.MoE)

    train_data = data_loader.DataSet(data=train_data,
                                     targets=train_targets,
                                     class_dict=config.CLASS_DICT)

    sampler = torch.utils.data.RandomSampler(data_source=train_data)
    train_loader = DataLoader(dataset=train_data,
                              batch_size=config.BATCH_SIZE,
                              sampler=sampler)

    model_train = util.setup_model(config=config,
                                   use_gpu=gpu)
    util.count_parameters(current_model=model_train,
                          logger=logger)

    learning_rate = config.LEARNING_RATE
    optimizer = optim.Adam(params=model_train.parameters(),
                           lr=learning_rate,
                           weight_decay=config.WEIGHT_DECAY)

    train_container = container.ResultsContainer(
        config=config,
        data_split="Train",
        use_gpu=gpu,
        use_weights=config.CLASS_WEIGHTS,
        class_weights=class_weights)
    test_container = container.ResultsContainer(
        config=config,
        data_split="Validation",
        use_gpu=gpu,
        use_weights=False,
        class_weights=None)

    if checkpoint and os.path.exists(os.path.join(config.EXP_DIR,
                                                  "data_info.pkl")):
        start_epoch, _ = checkpoint_data["epoch"]
        start_epoch += 1
        train_container.main_dict = checkpoint_data["train_container"]
        test_container.main_dict = checkpoint_data["test_container"]
        train_loader = checkpoint_data["train_loader"]
        util.load_model(checkpoint_path=os.path.join(config.EXP_DIR,
                                                     "model_info.pth"),
                        model=model_train,
                        cuda=torch.cuda.is_available(),
                        optimizer=optimizer)
    else:
        start_epoch = 0
    for cur_epoch in range(start_epoch, config.EPOCHS):
        tq = tqdm(total=len(train_targets))
        model_train.train()
        for data in train_loader:
            batch_data, batch_target = data

            if gpu:
                batch_data = batch_data.cuda()
                batch_target = batch_target.cuda()

            out = model_train(batch_data.unsqueeze(1))
            train_container.__updater__(inpt=out,
                                        label=(batch_target,))

            loss = train_container.__get_losses__()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            tq.update(config.BATCH_SIZE)

        tq.close()

        train_container.__end_epoch__()
        train_container.__printer__(epoch=cur_epoch)
        train_container.__logger__(logger=logger,
                                   epoch=cur_epoch)
        train_container.__resetter__()

        if cur_epoch > 0 and cur_epoch % 10 == 0:
            learning_rate = learning_rate / 10
            for param_group in optimizer.param_groups:
                param_group['lr'] = learning_rate

        best_scores_valid = run_eval(model_train=model_train,
                                     validation_data=validation_data,
                                     epoch=cur_epoch,
                                     model_path=MODEL_PATH,
                                     test_container=test_container)

    return best_scores_valid


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--debug', action='store_true', default=False,
                        help="Use flag to run in debug mode. Deletes "
                             "existing experiment dir.")
    parser.add_argument('--checkpoint', action='store_true', default=False,
                        help="Use flag to continue an unfinished experiment.")
    parser.add_argument('--gpu', action='store_true', default=False,
                        help="Parse this argument to use GPU while training")

    args = parser.parse_args()
    debug = args.debug
    checkpoint = args.checkpoint
    gpu = args.gpu

    num_folds = config.NUM_FOLDS
    seeds = config.SEEDS
    features_to_use = config.FEATURES_TO_USE

    total_scores_ua = {"main": []}
    total_scores_wa = {"main": []}
    total_scores_loss = {"main": []}
    if config.MoE and config.FUSION_LEVEL >= 0:
        for i in list(config.CLASS_DICT):
            total_scores_ua[i] = []
            total_scores_wa[i] = []
            total_scores_loss[i] = []

    if os.path.exists(config.EXP_DIR) and debug:
        shutil.rmtree(config.EXP_DIR, ignore_errors=False, onerror=None)
    if not os.path.exists(config.EXP_DIR):
        os.makedirs(config.EXP_DIR)
    else:
        if checkpoint:
            pass
        elif not config.SKIP_TRAIN:
            sys.exit(f"Experiment Dir Exists: {config.EXP_DIR}")

    features_exist = util.feature_checker(config=config)

    if checkpoint and os.path.exists(os.path.join(config.EXP_DIR,
                                                  "data_info.pkl")):
        checkpoint_data = util.load_checkpoint_info(
            pth=os.path.join(config.EXP_DIR, "data_info.pkl"))
        total_scores_ua = checkpoint_data["total_scores_ua"]
        total_scores_wa = checkpoint_data["total_scores_wa"]
        total_scores_loss = checkpoint_data["total_scores_loss"]

        start_epoch, start_fold = checkpoint_data["epoch"]
        start_seed = checkpoint_data["seed"]
        if start_epoch+1 == config.EPOCHS:
            if start_fold == config.NUM_FOLDS and start_fold != -1:
                if start_seed == len(config.SEEDS) - 1:
                    checkpoint = False
                    sys.exit("Reached the end of training anyway")
                else:
                    start_seed += 1
                    start_fold = start_epoch = 0
            elif start_fold < config.NUM_FOLDS and start_fold != -1:
                start_fold += 1
                start_epoch = 0
            else:
                if start_seed == len(config.SEEDS) - 1:
                    checkpoint = False
                    sys.exit("Reached the end of training anyway")
                else:
                    start_seed += 1
                    start_fold = start_epoch = 0
    else:
        start_epoch = start_fold = start_seed = 0

    if not config.SKIP_TRAIN:
        shutil.copyfile(config.__file__, config.EXP_DIR + "/config.py")
        for seed in range(start_seed, len(seeds)):
            exp_dir = os.path.join(config.EXP_DIR, f"seed_{seed}")
            if not os.path.exists(exp_dir):
                os.makedirs(exp_dir)
            util.setup_seed(seeds[seed])
            for exp_fold in range(start_fold, num_folds - 1):
                logger = util.make_logger(
                    log_name=os.path.join(exp_dir,
                                          f"train_{seed}_fold_{exp_fold}.log"),
                    config=config)

                MODEL_PATH = f"fold_{exp_fold}_seed_{seed}" \
                             f"_{config.FEATURES_TO_USE}.pth"
                MODEL_PATH = os.path.join(exp_dir, MODEL_PATH)

                dataset_dict = data_preprocessing(
                    config=config,
                    features_exist=features_exist,
                    exp_fold=exp_fold,
                    dataset=config.DATASET)

                features_exist = True

                _ = train_model(dataset_dict=dataset_dict,
                                logger=logger)
                checkpoint = False
                handlers = logger.handlers[:]
                for handler in handlers:
                    handler.close()
                    logger.removeHandler(handler)

            with torch.no_grad():
                best_scores = util.get_fold_score(config=config,
                                                  seed=seed,
                                                  model_path=MODEL_PATH,
                                                  use_gpu=gpu)
            total_scores_ua, total_scores_wa, total_scores_loss = \
                util.append_scores(config=config,
                                   total_scores_ua=total_scores_ua,
                                   total_scores_wa=total_scores_wa,
                                   total_scores_loss=total_scores_loss,
                                   best_scores=best_scores)

        util.print_res(config=config,
                       ua=total_scores_ua,
                       wa=total_scores_wa,
                       loss=total_scores_loss)

    else:
        logger = util.make_logger(
            log_name=os.path.join(config.EXP_DIR,
                                  f"{config.RUN_INFERENCE}_skipTrain.log"),
            config=config)
        with torch.no_grad():
            run_test(use_gpu=gpu,
                     score_containers=(total_scores_ua,
                                       total_scores_wa,
                                       total_scores_loss),
                     inference_type=config.RUN_INFERENCE)
