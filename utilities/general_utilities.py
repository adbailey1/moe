import sys

import torch
import numpy as np
import random
import logging
import logging.handlers
import os
import pickle

import container
from data_processes.data_processing import load_fold
import model


def get_class_weights(labels, emotions, use_class_weights=False, moe=False,
                      use_cuda=False):
    """
    Calculate the class weights according to the below formula. If
    experiment doesn't require class weights, create weights equal to "1"
    for all classes

    w_i = min(Class) / Class_i

    Args:
        labels: list - the target labels for the dataset
        use_class_weights: bool - set True to use class weights
        emotions: dict - key - int, value - emotion
        moe: bool - Set True if using the Mixture of Experts model
        use_cuda: bool - Set True to train with GPU

    Returns:

    """
    if use_class_weights:
        values = [0] * len(emotions)
        for label in labels:
            values[label] += 1

        summed_value = sum(values)
        min_value = min(values)
        weights = torch.tensor([min_value / i for i in values])

        if moe:
            moe_weights = []
            for emo_value in values:
                total_weight = emo_value / summed_value
                moe_weights.append(torch.tensor([total_weight, 1.]))
    else:
        weights = torch.ones(len(emotions))
        if moe:
            moe_weights = [torch.tensor([1., 1.])] * len(emotions)
    if use_cuda:
        weights = weights.cuda()
        if moe:
            moe_weights = [i.cuda() for i in moe_weights]

    return (weights,) if not moe else (weights, moe_weights)


def setup_seed(seed):
    """
    Set up the random number generators for reproducibility
    Args:
        seed: int - the seed to use for the random number generators

    Returns: None

    """
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def make_logger(log_name, config):
    """
    Create a logger to capture important experimental information
    Args:
        log_name: str - name for the current log file
        config: config file holding experimental information

    Returns:
        logger: logger - newly created logger
    """
    logfile = os.path.join(config.EXP_DIR, log_name)

    main_logger = logging.getLogger('MainLogger')
    main_logger.setLevel(logging.INFO)
    main_handler = logging.handlers.RotatingFileHandler(logfile)
    main_logger.addHandler(main_handler)

    return main_logger


def load_pickle(location):
    """ Load data from pickle file
    Args:
        location: str - location of pickled file

    Returns:
        loaded data

    """
    with open(location, "rb") as f:
        return pickle.load(f)


def save_pickle(location, data):
    """ Save data to pickle file
    Args:
        location: str - location to save the data
        data: the data to be saved

    Returns: None

    """
    with open(location, "wb") as f:
        pickle.dump(data, f, pickle.HIGHEST_PROTOCOL)


def divider_checker(a, b):
    """ Checks that division is possible between 2 numbers

    Args:
        a: int - a number
        b: int - a number

    Returns:
        division of a and b as long as b is not 0
    """
    return a/b if b else -1.


def calc_ua_wa(matrix, num_correct, len_feats_dict, classes=4):
    """
    Calculates the unweighted accuracy and weighted accuracy according
    to the paper: LINK

    UA = 1/N Sum_c=1^K ((TP_c) / (TP_c + FN_c))
    WA = Sum (TP / (TP + FN))

    Args:
        matrix: numpy.matrix - holds the ground truths vs predictions
        num_correct: int - running total of number of correct predictions
        len_feats_dict: int - counter of instances in the current data set
        classes: int - how many classes are being classified

    Returns:
        ua: float - the unweighted accuracy
        wa: float - the weighted accuracy

    """
    ua = [0] * classes
    class_total = matrix.sum(1).flatten().tolist()[0]
    for row in range(classes):
        ua[row] = round(divider_checker(a=matrix[row, row],
                                        b=class_total[row]), 3)
    wa = num_correct / len_feats_dict
    if -1 in ua:
        del ua[ua.index(-1)]

    return ua, wa


def forward_pass_test_data(model, validation_data, test_container,
                           use_gpu=False):
    """
    Pass the validation data through the model and update the container
    with the predictions

    Args:
        model: model to pass data through
        validation_data: dict - the validation data, keys: data (numpy.array)
            target (numpy.array)
        test_container: container module - holds all prediction data for the
            validation set
        use_gpu: Bool - set True to train with GPU

    Returns: None

    """
    model.eval()
    with torch.no_grad():
        for key in validation_data:
            batch_data, batch_target = validation_data[key]["data"], \
                                       validation_data[key]["targets"]
            batch_data = torch.from_numpy(batch_data).float()
            batch_target = torch.LongTensor([batch_target[0]])

            if use_gpu:
                batch_data = batch_data.cuda()
                batch_target = batch_target.cuda()

            out = model(batch_data.unsqueeze(1))

            test_container.__updater__(out, (batch_target,), squash=True)


def calc_num_correct(pred, target, classes, squash=False, key="main"):
    """
    Creates a matrix of ground truth vs predictions and accumulates the
    total number of correct predictions

    Args:
        pred: torch.tensor - prediction outputs from training model
        target: torch.tensor - respective labels for the predictions
        classes: int - number of classes in the dataset
        squash: bool - set True when using validation set to average the
            predicted results
        key: str - the key for the prediction container dictionary (main,
            neu, hap, sad, ang)

    Returns:
        matrix: numpy.matrix - ground truth vs predictions for current model
            output
        num_correct: int - the total number of correct predictions from the
            current model output
    """
    matrix = np.matrix(np.zeros((classes, classes)), dtype=int)
    num_correct = 0
    if key != "main":
        pred = torch.round(pred)
    else:
        if squash:
            pred = torch.argmax(pred).view(1, -1)
        else:
            pred = torch.argmax(pred, dim=1)

    for i, p in enumerate(pred):
        if p == target[i]:
            num_correct = num_correct + 1
        # Ground Truth Vertical, Prediction Horizontal
        matrix[int(target[i]), int(p)] += 1

    return matrix, num_correct


def setup_model(config, use_gpu=False):
    """ Creates the model to be used in training or testing

    Args:
        config: Hold experimental information
        use_gpu: Bool - set True to train with a GPU

    Returns:
        model_train: the model to be used in training or testing
    """
    if config.MODEL_TYPE == "MACNN":
        if config.MoE:
            if config.FUSION_LEVEL == 0:
                model_train = model.MACNNMixtureOfExperts(
                    attention_heads=config.ATTENTION_HEADS,
                    attention_hidden=config.ATTENTION_HIDDEN)
            elif config.FUSION_LEVEL == 1:
                model_train = model.MACNNMixtureOfExpertsForwardFusion1(
                    attention_heads=config.ATTENTION_HEADS,
                    attention_hidden=config.ATTENTION_HIDDEN,
                    fusion_level=config.FUSION_LEVEL)
            elif config.FUSION_LEVEL == 2:
                model_train = model.MACNNMixtureOfExpertsForwardFusion2(
                    attention_heads=config.ATTENTION_HEADS,
                    attention_hidden=config.ATTENTION_HIDDEN,
                    fusion_level=config.FUSION_LEVEL)
            elif config.FUSION_LEVEL == 3:
                model_train = model.MACNNMixtureOfExpertsForwardFusion3(
                    attention_heads=config.ATTENTION_HEADS,
                    attention_hidden=config.ATTENTION_HIDDEN,
                    fusion_level=config.FUSION_LEVEL)
            elif config.FUSION_LEVEL == 4:
                model_train = model.MACNNMixtureOfExpertsForwardFusion4(
                    attention_heads=config.ATTENTION_HEADS,
                    attention_hidden=config.ATTENTION_HIDDEN,
                    fusion_level=config.FUSION_LEVEL)
            elif config.FUSION_LEVEL == 5:
                model_train = model.MACNNMixtureOfExpertsForwardFusion5(
                    attention_heads=config.ATTENTION_HEADS,
                    attention_hidden=config.ATTENTION_HIDDEN,
                    fusion_level=config.FUSION_LEVEL)
            elif config.FUSION_LEVEL == 6:
                model_train = model.MACNNMixtureOfExpertsForwardFusion6(
                    attention_heads=config.ATTENTION_HEADS,
                    attention_hidden=config.ATTENTION_HIDDEN,
                    fusion_level=config.FUSION_LEVEL)
            elif config.FUSION_LEVEL == -7:
                model_train = model.MACNNMixtureOfExpertsBackwardFusion7(
                    attention_heads=config.ATTENTION_HEADS,
                    attention_hidden=config.ATTENTION_HIDDEN,
                    num_moe=4)
            elif config.FUSION_LEVEL == -6:
                model_train = model.MACNNMixtureOfExpertsBackwardFusion6(
                    attention_heads=config.ATTENTION_HEADS,
                    attention_hidden=config.ATTENTION_HIDDEN,
                    fusion_level=config.FUSION_LEVEL,
                    num_moe=4)
            elif config.FUSION_LEVEL == -5:
                model_train = model.MACNNMixtureOfExpertsBackwardFusion5(
                    attention_heads=config.ATTENTION_HEADS,
                    attention_hidden=config.ATTENTION_HIDDEN,
                    fusion_level=config.FUSION_LEVEL,
                    num_moe=4)
            elif config.FUSION_LEVEL == -4:
                model_train = model.MACNNMixtureOfExpertsBackwardFusion4(
                    attention_heads=config.ATTENTION_HEADS,
                    attention_hidden=config.ATTENTION_HIDDEN,
                    fusion_level=config.FUSION_LEVEL)
            elif config.FUSION_LEVEL == -3:
                model_train = model.MACNNMixtureOfExpertsBackwardFusion3(
                    attention_heads=config.ATTENTION_HEADS,
                    attention_hidden=config.ATTENTION_HIDDEN,
                    fusion_level=config.FUSION_LEVEL,
                    num_moe=4)
            elif config.FUSION_LEVEL == -2:
                model_train = model.MACNNMixtureOfExpertsBackwardFusion2(
                    attention_heads=config.ATTENTION_HEADS,
                    attention_hidden=config.ATTENTION_HIDDEN,
                    fusion_level=config.FUSION_LEVEL,
                    num_moe=4)
        else:
            model_train = model.MACNN(
                attention_heads=config.ATTENTION_HEADS,
                attention_hidden=config.ATTENTION_HIDDEN)
    elif config.MODEL_TYPE == "MACNN_x4":
        model_train = model.MACNN4timesParams(
            attention_heads=config.ATTENTION_HEADS,
            attention_hidden=config.ATTENTION_HIDDEN)
    else:
        raise ValueError(f"Wrong model selected in config: MACNN or MACNN_x4")

    if use_gpu:
        model_train = model_train.cuda()
    return model_train


def count_parameters(current_model, logger):
    """ Counts the number of trainable parameters in the model
    Args:
        current_model: the model to count
        logger: logger to capture important information

    Returns: None

    """
    model_params = sum(params.numel() for params in current_model.parameters()
                       if params.requires_grad)
    print(f"Number of Params: {model_params}")
    logger.info(f"Number of Params: {model_params}")


def print_res(config, ua, wa, loss, data_type="Validation"):
    """
    Prints and logs the results from training in terms of unweighted accuracy,
    weighted accuracy, and loss

    Args:
        config: config file holding experimental information
        ua: float - the unweighted accuracy result
        wa: float - the weighted accuracy result
        loss: float - the loss value
        data_type: str - set to "Validation" for validation set or "Test"
            for test set

    Returns: None

    """
    print(f"Total UA: {ua}")
    print(f"Total WA: {wa}")
    print(f"Total Loss: {loss}")
    logger = make_logger(
        log_name=os.path.join(config.EXP_DIR, f"{data_type}.log"),
        config=config)
    logger.info(f"Total UA: {ua}")
    logger.info(f"Total WA: {wa}")
    logger.info(f"Total Loss: {loss}")

    print(f'Emotion Average UA: {np.round(np.mean(ua["main"]), 4)}')
    print(f'Emotion Average WA: {np.round(np.mean(wa["main"]), 4)}')
    print(f'Emotion Average Loss: {np.round(np.mean(loss["main"]), 4)}')

    if config.MoE:
        emos_to_use = list(config.CLASS_DICT)
        if config.FUSION_LEVEL > -1:
            print(f'Average UA ({emos_to_use}):'
                  f' {np.round(np.mean(ua[emos_to_use[0]]), 4)}, '
                  f'{np.round(np.mean(ua[emos_to_use[1]]), 4)}, '
                  f'{np.round(np.mean(ua[emos_to_use[2]]), 4)}, '
                  f'{np.round(np.mean(ua[emos_to_use[3]]), 4)}')
            print(f'Average WA ({emos_to_use}): '
                  f'{np.round(np.mean(wa[emos_to_use[0]]), 4)}, '
                  f'{np.round(np.mean(wa[emos_to_use[1]]), 4)}, '
                  f'{np.round(np.mean(wa[emos_to_use[2]]), 4)}, '
                  f'{np.round(np.mean(wa[emos_to_use[3]]), 4)}')
            print(f'Average Loss ({emos_to_use}): '
                  f'{np.round(np.mean(loss[emos_to_use[0]]), 4)}, '
                  f'{np.round(np.mean(loss[emos_to_use[1]]), 4)}, '
                  f'{np.round(np.mean(loss[emos_to_use[2]]), 4)}, '
                  f'{np.round(np.mean(loss[emos_to_use[3]]), 4)}')

    logger.info(f'Emotion Average UA: {np.round(np.mean(ua["main"]), 4)}')
    logger.info(f'Emotion Average WA: {np.round(np.mean(wa["main"]), 4)}')
    logger.info(f'Emotion Average Loss:'
                f' {np.round(np.mean(loss["main"]), 4)}')

    if config.MoE:
        if config.FUSION_LEVEL > -1:
            logger.info(f'Average UA ({emos_to_use}):'
                        f' {np.round(np.mean(ua[emos_to_use[0]]), 4)}, '
                        f'{np.round(np.mean(ua[emos_to_use[1]]), 4)},'
                        f'{np.round(np.mean(ua[emos_to_use[2]]), 4)},'
                        f'{np.round(np.mean(ua[emos_to_use[3]]), 4)}')
            logger.info(f'Average WA ({emos_to_use}): '
                        f'{np.round(np.mean(wa[emos_to_use[0]]), 4)}, '
                        f'{np.round(np.mean(wa[emos_to_use[1]]), 4)},'
                        f'{np.round(np.mean(wa[emos_to_use[2]]), 4)},'
                        f'{np.round(np.mean(wa[emos_to_use[3]]), 4)}')
            logger.info(f'Average Loss ({emos_to_use}): '
                        f'{np.round(np.mean(loss[emos_to_use[0]]), 4)}, '
                        f'{np.round(np.mean(loss[emos_to_use[1]]), 4)}, '
                        f'{np.round(np.mean(loss[emos_to_use[2]]), 4)}, '
                        f'{np.round(np.mean(loss[emos_to_use[3]]), 4)}')


def feature_checker(config):
    """ Check if the data folds exists

    Args:
        config: config file holding experimental information

    Returns:
        features_exist: bool - set True if data folds exist

    """
    if not os.path.exists(os.path.join(config.FEATURE_LOC,
                                       f"fold_0_{config.FOLD_FILENAME}")):
        features_exist = False
    else:
        features_exist = True

    return features_exist


def append_scores(config, total_scores_ua, total_scores_wa,
                  total_scores_loss, best_scores):
    """
    Append the current results (unweighted / weighted accuracy and loss) to
    the results dictionaries

    Args:
        config: config file holding experimental information
        total_scores_ua: dictionary keys: main (opt: neu, hap, sad, ang)
        total_scores_wa: dictionary keys: main (opt: neu, hap, sad, ang)
        total_scores_loss: dictionary keys: main (opt: neu, hap, sad, ang)
        best_scores: dictionary from container

    Returns:
        total_scores_ua: dict - update holder for unweighted accuracy
        total_scores_wa: dict - update holder for weighted accuracy
        total_scores_loss dict - update holder for loss
    """
    total_scores_ua["main"].append(best_scores["max_ua"]["main"])
    total_scores_wa["main"].append(best_scores["max_wa"]["main"])
    total_scores_loss["main"].append(best_scores["best_loss"]["main"])

    if config.MoE:
        emos_to_use = list(config.CLASS_DICT)
        if config.FUSION_LEVEL > -1:
            total_scores_ua[emos_to_use[0]].append(
                best_scores["max_ua"][emos_to_use[0]])
            total_scores_wa[emos_to_use[0]].append(
                best_scores["max_wa"][emos_to_use[0]])
            total_scores_loss[emos_to_use[0]].append(
                best_scores["best_loss"][emos_to_use[0]])

            total_scores_ua[emos_to_use[1]].append(
                best_scores["max_ua"][emos_to_use[1]])
            total_scores_wa[emos_to_use[1]].append(
                best_scores["max_wa"][emos_to_use[1]])
            total_scores_loss[emos_to_use[1]].append(
                best_scores["best_loss"][emos_to_use[1]])

            total_scores_ua[emos_to_use[2]].append(
                best_scores["max_ua"][emos_to_use[2]])
            total_scores_wa[emos_to_use[2]].append(
                best_scores["max_wa"][emos_to_use[2]])
            total_scores_loss[emos_to_use[2]].append(
                best_scores["best_loss"][emos_to_use[2]])

            total_scores_ua[emos_to_use[3]].append(
                best_scores["max_ua"][emos_to_use[3]])
            total_scores_wa[emos_to_use[3]].append(
                best_scores["max_wa"][emos_to_use[3]])
            total_scores_loss[emos_to_use[3]].append(
                best_scores["best_loss"][emos_to_use[3]])

    return total_scores_ua, total_scores_wa, total_scores_loss


def save_info_for_checkpoint(seed, epoch, train_container, test_container,
                             save_dir, model, optimizer, cuda, train_loader,
                             total_scores_ua, total_scores_wa,
                             total_scores_loss):
    """ Saves current experiment state, models, and data in case program
    is stopped and does not finish the current experiment.

    Args:
        seed: int - current seed for the random number generators
        epoch: int - the current experiment epoch
        train_container: container holding experimental data for training set
        test_container: container holding experimental data for validation set
        save_dir: str - location to save the checkpoint information
        model: the experiment model
        optimizer: current optimiser and status for experiment
        cuda: torch.is_cuda_available()
        train_loader: PyTorch Dataloader
        total_scores_ua: dictionary keys: main (opt: neu, hap, sad, ang)
        total_scores_wa: dictionary keys: main (opt: neu, hap, sad, ang)
        total_scores_loss: dictionary keys: main (opt: neu, hap, sad, ang)

    Returns: None

    """
    save_location_model_data = os.path.join(save_dir, "model_info.pth")
    save_location_data = os.path.join(save_dir, "data_info.pkl")
    save_out_dict = {'epoch': epoch,
                     'state_dict': model.state_dict(),
                     'optimizer': optimizer.state_dict(),
                     'rng_state': torch.get_rng_state(),
                     'numpy_rng_state': np.random.get_state(),
                     'random_rng_state': random.getstate()}
    if cuda:
        save_out_dict['cuda_rng_state'] = torch.cuda.get_rng_state()

    save_data_dict = {"epoch": epoch,
                      "seed": seed,
                      "train_container": train_container,
                      "test_container": test_container,
                      "train_loader": train_loader,
                      "total_scores_ua": total_scores_ua,
                      "total_scores_wa": total_scores_wa,
                      "total_scores_loss": total_scores_loss}

    torch.save(save_out_dict, save_location_model_data)
    with open(save_location_data, "wb") as f:
        pickle.dump(save_data_dict, f)


def load_checkpoint_info(pth):
    """ Load data from unfinished experiment

    Args:
        pth: str - location of the checkpoint data

    Returns:
        checkpoint_data: dict - containing experimental data

    """
    with open(pth, "rb") as f:
        checkpoint_data = pickle.load(f)

    return checkpoint_data


def load_model(checkpoint_path, model, cuda, optimizer=None):
    """
    Loads the model weights along with the current epoch and all the random
    states that are used during the experiment. Also loads the current state
    of the data loader for continuity
    Inputs:
        checkpoint_path: Location of the saved model
        model: The model from current experiment
        optimizer: The current optimiser state
        cuda: bool - Set True to use GPU (set in initial arguments)
    Outputs:
        epoch_iter: Current epoch
        data_saver: Holds information regarding the data loader so that it
            can be restored from a checkpoint. This includes the
            current pointer of ones and zeros and the current list of
            indexes of the ones and zeros
    """
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['state_dict'], strict=False)
    if optimizer is not None:
        optimizer.load_state_dict(checkpoint['optimizer'])
    torch.set_rng_state(checkpoint['rng_state'])
    if cuda:
        torch.cuda.set_rng_state(checkpoint['cuda_rng_state'])
    np.random.set_state(checkpoint['numpy_rng_state'])
    random.setstate(checkpoint['random_rng_state'])


def get_fold_score(config, seed, model_path, use_gpu=False):
    """ Load models and obtain results with validation or test data

    Args:
        config: config file holding experimental information
        seed: int - current seed of the experiment
        model_path: str - base path to the models to load
        use_gpu: Bool - set True to train with GPU

    Returns:
        current_best_scores: dict containing the best unweighted / weighted
            accuracy, matrix, and loss scores

    """
    test_container_wa = container.ResultsContainer(config=config,
                                                   use_gpu=use_gpu,
                                                   data_split="Validation",
                                                   use_weights=False,
                                                   class_weights=None)
    test_container_ua = container.ResultsContainer(config=config,
                                                   use_gpu=use_gpu,
                                                   data_split="Validation",
                                                   use_weights=False,
                                                   class_weights=None)
    tc = [test_container_wa, test_container_ua]
    for exp_fold in range(config.NUM_FOLDS - 1):
        fold_data = load_fold(config=config,
                              fold=exp_fold)
        test_data = fold_data["valid_data"]

        MODEL_PATH_WA = f"fold_{exp_fold}_seed_{seed}" \
                        f"_{config.FEATURES_TO_USE}_wa.pth"
        MODEL_PATH_WA = os.path.join(config.EXP_DIR, f"seed_{seed}",
                                     MODEL_PATH_WA)
        MODEL_PATH_UA = f"fold_{exp_fold}_seed_{seed}" \
                        f"_{config.FEATURES_TO_USE}_ua.pth"
        MODEL_PATH_UA = os.path.join(config.EXP_DIR, f"seed_{seed}",
                                     MODEL_PATH_UA)

        test_model = setup_model(config, use_gpu=False)
        for p, m in enumerate([MODEL_PATH_WA, MODEL_PATH_UA]):
            test_model.load_state_dict(torch.load(m))

            if use_gpu:
                test_model = test_model.cuda()
            test_model.eval()

            forward_pass_test_data(model=test_model,
                                   validation_data=test_data,
                                   test_container=tc[p],
                                   use_gpu=use_gpu)

    for p, m in enumerate([MODEL_PATH_WA, MODEL_PATH_UA]):
        tc[p].__end_epoch__(save_model=False,
                            model_to_save=(test_model, model_path))

        temp_scores = tc[p].__getter__()
        if p == 0:
            current_best_scores = temp_scores
        else:
            current_best_scores["max_ua"] = temp_scores["max_ua"]

    return current_best_scores
