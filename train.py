# https://github.com/lessonxmk/head_fusion
import os
import pickle
import sys
import argparse
import shutil

import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from configs import config
from data_processes import data_loader
from utilities import general_utilities as util
from data_processes.data_processing_ravdess import load_fold
import container


def run_test(inference_type="Test", results_container_main_key="main"):
    total_test_ua = {"main": [], "spkr": [],
                     "neu": [], "hap": [], "sad": [], "ang": []}
    total_test_wa = {"main": [], "spkr": [],
                     "neu": [], "hap": [], "sad": [], "ang": []}
    total_test_f1 = {"main": [], "spkr": [],
                     "neu": [], "hap": [], "sad": [], "ang": []}
    total_test_loss_wa = {"main": [], "spkr": [],
                          "neu": [], "hap": [], "sad": [], "ang": []}
    total_test_loss_f1 = {"main": [], "spkr": [],
                          "neu": [], "hap": [], "sad": [], "ang": []}

    for seed in range(len(seeds)):
        if inference_type == "Validation":
            test_container_wa = container.ResultsContainer(config)
            test_container_ua = container.ResultsContainer(config)
            test_container_f1 = container.ResultsContainer(config)
            tc = [test_container_wa, test_container_ua, test_container_f1]
        for exp_fold in range(config.NUM_FOLDS - 1):
            if inference_type != "Validation":
                test_container_wa = container.ResultsContainer(config)
                test_container_ua = container.ResultsContainer(config)
                test_container_f1 = container.ResultsContainer(config)
                tc = [test_container_wa, test_container_ua,
                      test_container_f1]

            feat_dict = load_fold(config, exp_fold)
            data = feat_dict["val_dict"]

            MODEL_PATH_WA = f"fold_{exp_fold}_seed_{seed}" \
                            f"_{config.FEATURES_TO_USE}_wa.pth"
            MODEL_PATH_WA = os.path.join(config.EXP_DIR, f"seed_{seed}",
                                         MODEL_PATH_WA)
            MODEL_PATH_UA = f"fold_{exp_fold}_seed_{seed}" \
                            f"_{config.FEATURES_TO_USE}_ua.pth"
            MODEL_PATH_UA = os.path.join(config.EXP_DIR, f"seed_{seed}",
                                         MODEL_PATH_UA)
            MODEL_PATH_F1 = f"fold_{exp_fold}_seed_{seed}" \
                            f"_{config.FEATURES_TO_USE}_f1.pth"
            MODEL_PATH_F1 = os.path.join(config.EXP_DIR, f"seed_{seed}",
                                         MODEL_PATH_F1)

            test_model = util.setup_model(config)
            for p, m in enumerate(
                    [MODEL_PATH_WA, MODEL_PATH_UA, MODEL_PATH_F1]):
                if not config.MIXTUREOFEXPERTS:
                    test_model = util.setup_model(config)
                    test_model.load_state_dict(torch.load(m))
                else:
                    if os.path.exists(m.replace(".pth", "N.pth")):
                        if (not config.SKIP_FINAL_FC and not
                        config.MIXTUREOFEXPERTS and config.FUSION_LEVEL > -1):
                            test_model.load_state_dict(torch.load(m))
                        test_model.mod_neutral.load_state_dict(
                            torch.load(m.replace(".pth", "N.pth")))
                        test_model.mod_happy.load_state_dict(
                            torch.load(m.replace(".pth", "H.pth")))
                        test_model.mod_sad.load_state_dict(
                            torch.load(m.replace(".pth", "S.pth")))
                        test_model.mod_angry.load_state_dict(
                            torch.load(m.replace(".pth", "A.pth")))
                    else:
                        test_model = util.setup_model(config)
                        test_model.load_state_dict(torch.load(m))

                if torch.cuda.is_available():
                    test_model = test_model.cuda()
                test_model.eval()

                mapper = None

                util.forward_pass_test_data_iemocap(config=config,
                                                    model_train=test_model,
                                                    feat_dict=data,
                                                    test_container=tc[p],
                                                    mapper=mapper)
            if inference_type != "Validation":
                for p, m in enumerate(
                        [MODEL_PATH_WA, MODEL_PATH_UA, MODEL_PATH_F1]):
                    tc[p].__end_epoch__(save_model=False,
                                        model_to_save=(test_model, MODEL_PATH_WA))

                    temp_scores = tc[p].__getter__()
                    if p == 0:
                        best_scores = temp_scores
                    elif p == 1:
                        best_scores["maxUA"] = temp_scores["maxUA"]
                    else:
                        best_scores["maxF1"] = temp_scores["maxF1"]

                total_test_ua, total_test_wa, total_test_f1, \
                total_test_loss_wa, total_test_loss_f1 = util.append_scores(
                    config, total_test_ua, total_test_wa,
                    total_test_f1, total_test_loss_wa,
                    total_test_loss_f1, best_scores)

        if inference_type == "Validation":
            for p, m in enumerate(
                    [MODEL_PATH_WA, MODEL_PATH_UA, MODEL_PATH_F1]):
                tc[p].__end_epoch__(save_model=False,
                                    model_to_save=(
                                    test_model, MODEL_PATH_WA))

                temp_scores = tc[p].__getter__()
                if p == 0:
                    best_scores = temp_scores
                elif p == 1:
                    best_scores["maxUA"] = temp_scores["maxUA"]
                else:
                    best_scores["maxF1"] = temp_scores["maxF1"]

            total_test_ua, total_test_wa, total_test_f1, \
            total_test_loss_wa, total_test_loss_f1 = util.append_scores(
                config, total_test_ua, total_test_wa,
                total_test_f1, total_test_loss_wa,
                total_test_loss_f1, best_scores)


    util.print_res(config, total_test_ua, total_test_wa,
                   total_test_f1, total_test_loss_wa,
                   total_test_loss_f1, inference_type)


def run_eval(model_train, feat_dict, epoch, model_path, test_container,
             save_model=True):
    # TODO check that this works. Need to write comments as I don't know why
    #  we wouldn't want a mapper in the speeaker independent setup. What
    #  happens when we use sessions 1, 2, 4 as training?
    mapper = None

    util.forward_pass_test_data_iemocap(config, model_train, feat_dict,
                                        test_container, mapper)
    test_container.__end_epoch__(save_model=save_model,
                                 model_to_save=(model_train, model_path))
    test_container.__printer__(epoch)
    test_container.__logger__(logger, epoch)
    best_scores = test_container.__getter__()
    test_container.__reseter__()

    return best_scores


def train_model(feat_dict, logger, results_container_main_key="main"):
    train_data_features = feat_dict["train_data"]
    train_label = feat_dict["train_labels"]
    if "train_spkr_id" in feat_dict:
        train_spkr_id = feat_dict["train_spkr_id"]
    valid_features_dict = feat_dict["val_dict"]
    learning_rate = config.LEARNING_RATE
    print(learning_rate)
    for cur_label in train_label:
        config.LABEL_NUM[cur_label] += 1
    if config.MIXTUREOFEXPERTS:
        weight, moe_weight = util.calc_weight(config)
    else:
        weight = util.calc_weight(config)

    if "train_spkr_id" in feat_dict:
        train_data = data_loader.DataSet(config=config,
                                         data=train_data_features,
                                         labels=train_label,
                                         spkr_id=train_spkr_id)
    else:
        train_data = data_loader.DataSet(config=config,
                                         data=train_data_features,
                                         labels=train_label)

    sampler = torch.utils.data.RandomSampler(data_source=train_data)
    train_loader = DataLoader(train_data, batch_size=config.BATCH_SIZE,
                              sampler=sampler)

    model_train = util.setup_model(config)
    model_params = util.count_parameters(model_train)
    print(f"Number of Params: {model_params}")
    logger.info(f"Number of Params: {model_params}")
    if torch.cuda.is_available():
        model_train = model_train.cuda()

    optimizer = optim.Adam(model_train.parameters(),
                           lr=learning_rate,
                           weight_decay=config.WEIGHT_DECAY)

    logger.info("training...")

    if config.USE_WEIGHTS_FOR_LOSS:
        if config.MIXTUREOFEXPERTS:
            train_container = container.ResultsContainer(
                config, "Train", main_key=results_container_main_key,
                weights=(weight, moe_weight))
        else:
            train_container = container.ResultsContainer(
                config, "Train", main_key=results_container_main_key,
                weights=(weight,))
    else:
        train_container = container.ResultsContainer(
            config, "Train", main_key=results_container_main_key)
    test_container = container.ResultsContainer(
        config, "Validation", main_key=results_container_main_key)

    if checkpoint and os.path.exists(os.path.join(config.EXP_DIR,
                                                  "data_info.pkl")):
        start_epoch, _ = checkpoint_data["epoch"]
        start_epoch += 1
        train_container.main_dict = checkpoint_data["train_container"]
        test_container.main_dict = checkpoint_data["test_container"]
        train_loader = checkpoint_data["train_loader"]

        optimizers = (optimizer,)
        util.load_model(os.path.join(config.EXP_DIR, "model_info.pth"),
                        model_train, torch.cuda.is_available(), optimizers)
    else:
        start_epoch = 0
    for cur_epoch in range(start_epoch, config.EPOCHS):
        tq = tqdm(total=len(train_label))
        model_train.train()
        for data in train_loader:
            if "train_spkr_id" in feat_dict:
                x, y, s = data
                if torch.cuda.is_available():
                    x = x.cuda()
                    y = y.cuda()
                    s = s.cuda()

                out = model_train(x.unsqueeze(1))
                train_container.__updater__(out, (y, s))
            else:
                x, y = data
                if torch.cuda.is_available():
                    x = x.cuda()
                    y = y.cuda()

                out = model_train(x.unsqueeze(1))
                train_container.__updater__(out, (y,))

            loss = train_container.__get_losses__()
            if config.MODEL_TYPE == "LightSERNet":
                l2_lambda = 1e-6
                l2_norm = 0
                if config.MIXTUREOFEXPERTS:
                    for mod in [model_train.mod_neutral, model_train.mod_happy,
                                model_train.mod_sad, model_train.mod_angry]:
                        l2_norm = l2_norm + torch.linalg.norm(mod.conv2.weight)
                        l2_norm = l2_norm + torch.linalg.norm(mod.conv3.weight)
                        l2_norm = l2_norm + torch.linalg.norm(mod.conv4.weight)
                        l2_norm = l2_norm + torch.linalg.norm(mod.conv5.weight)
                        l2_norm = l2_norm + torch.linalg.norm(mod.conv6.weight)
                else:
                    l2_norm = l2_norm + torch.linalg.norm(
                        model_train.conv2.weight)
                    l2_norm = l2_norm + torch.linalg.norm(
                        model_train.conv3.weight)
                    l2_norm = l2_norm + torch.linalg.norm(
                        model_train.conv4.weight)
                    l2_norm = l2_norm + torch.linalg.norm(
                        model_train.conv5.weight)
                    l2_norm = l2_norm + torch.linalg.norm(
                        model_train.conv6.weight)

                loss = loss + l2_lambda * l2_norm
            optimizer.zero_grad()
            loss.backward(retain_graph=True)
            optimizer.step()
            tq.update(config.BATCH_SIZE)

        tq.close()

        train_container.__end_epoch__()
        train_container.__printer__(cur_epoch)
        train_container.__logger__(logger, cur_epoch)
        train_container.__reseter__()

        if config.MODEL_TYPE == "LightSERNet":
            if cur_epoch > 49 and cur_epoch % 20 == 0:
                learning_rate = learning_rate * np.exp(-.15)
        else:
            if cur_epoch > 0 and cur_epoch % 10 == 0:
                learning_rate = learning_rate / 10
                for param_group in optimizer.param_groups:
                    param_group['lr'] = learning_rate

        best_scores_valid = run_eval(model_train=model_train,
                                     feat_dict=valid_features_dict,
                                     epoch=cur_epoch, model_path=MODEL_PATH,
                                     test_container=test_container)
        optimizers = (optimizer,)

        util.save_info_for_checkpoint(seed, (cur_epoch, exp_fold),
                                      train_container.main_dict,
                                      test_container.main_dict,
                                      config.EXP_DIR, model_train, optimizers,
                                      torch.cuda.is_available(), train_loader,
                                      total_scores_ua, total_scores_wa,
                                      total_scores_f1,
                                      total_scores_loss_wa,
                                      total_scores_loss_f1
                                      )

    return best_scores_valid


def get_fold_score(seed):
    # comp_matrix = np.mat(np.zeros((4, 4)), dtype=int)
    # comp_num_correct = comp_len_feat_dict = 0
    test_container_wa = container.ResultsContainer(config)
    test_container_ua = container.ResultsContainer(config)
    test_container_f1 = container.ResultsContainer(config)
    tc = [test_container_wa, test_container_ua, test_container_f1]
    for exp_fold in range(config.NUM_FOLDS - 1):
        feat_dict = load_fold(config, exp_fold)
        data = feat_dict["val_dict"]

        MODEL_PATH_WA = f"fold_{exp_fold}_seed_{seed}" \
                        f"_{config.FEATURES_TO_USE}.pth"
        MODEL_PATH_WA = os.path.join(config.EXP_DIR, f"seed_{seed}",
                                     MODEL_PATH_WA)
        MODEL_PATH_UA = f"fold_{exp_fold}_seed_{seed}" \
                        f"_{config.FEATURES_TO_USE}_ua.pth"
        MODEL_PATH_UA = os.path.join(config.EXP_DIR, f"seed_{seed}",
                                     MODEL_PATH_UA)
        MODEL_PATH_F1 = f"fold_{exp_fold}_seed_{seed}" \
                        f"_{config.FEATURES_TO_USE}_f1.pth"
        MODEL_PATH_F1 = os.path.join(config.EXP_DIR, f"seed_{seed}",
                                     MODEL_PATH_F1)

        test_model = util.setup_model(config)
        for p, m in enumerate([MODEL_PATH_WA, MODEL_PATH_UA, MODEL_PATH_F1]):
            test_model.load_state_dict(torch.load(m))

            if torch.cuda.is_available():
                test_model = test_model.cuda()
            test_model.eval()

            if config.MTL and not config.SPEAKER_IND:
                save_map = MODEL_PATH[:-4] + "_mapper.pkl"
                with open(save_map, "rb") as f:
                    mapper = pickle.load(f)
            else:
                mapper = None

            util.forward_pass_test_data_iemocap(config=config,
                                                model_train=test_model,
                                                feat_dict=data,
                                                test_container=tc[p],
                                                mapper=mapper)
    for p, m in enumerate([MODEL_PATH_WA, MODEL_PATH_UA, MODEL_PATH_F1]):
        tc[p].__end_epoch__(save_model=False,
                            model_to_save=(test_model, MODEL_PATH))

        temp_scores = tc[p].__getter__()
        if p == 0:
            current_best_scores = temp_scores
        elif p == 1:
            current_best_scores["maxUA"] = temp_scores["maxUA"]
        else:
            current_best_scores["maxF1"] = temp_scores["maxF1"]
    return current_best_scores


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--debug', action='store_true', default=False,
                        help="Use flag to run in debug mode. Deletes "
                             "existing experiment dir.")
    parser.add_argument('--checkpoint', action='store_true', default=False,
                        help="Use flag to continue an unfinished experiment.")

    args = parser.parse_args()
    debug = args.debug
    checkpoint = args.checkpoint

    num_folds = config.NUM_FOLDS
    seeds = config.SEEDS
    features_to_use = config.FEATURES_TO_USE
    total_scores_ua = {"main": [], "gen": [], "spkr": [], "neu": [],
                       "hap": [], "sad": [], "ang": []}
    total_scores_wa = {"main": [], "gen": [], "spkr": [], "neu": [],
                       "hap": [], "sad": [], "ang": []}
    total_scores_f1 = {"main": [], "gen": [], "spkr": [], "neu": [],
                       "hap": [], "sad": [], "ang": []}
    total_scores_loss_wa = {"main": [], "gen": [], "spkr": [], "encdec": [],
                            "neu": [], "hap": [], "sad": [], "ang": []}
    total_scores_loss_f1 = {"main": [], "gen": [], "spkr": [], "encdec": [],
                            "neu": [], "hap": [], "sad": [], "ang": []}

    if os.path.exists(config.EXP_DIR) and debug:
        shutil.rmtree(config.EXP_DIR, ignore_errors=False, onerror=None)
    if not os.path.exists(config.EXP_DIR):
        os.makedirs(config.EXP_DIR)
    else:
        if checkpoint:
            pass
        elif not config.SKIP_TRAIN:
            sys.exit(f"Experiment Dir Exists: {config.EXP_DIR}")

    features_exist = util.feature_checker(config)

    if checkpoint and os.path.exists(os.path.join(config.EXP_DIR,
                                                  "data_info.pkl")):
        checkpoint_data = util.load_checkpoint_info(os.path.join(
            config.EXP_DIR, "data_info.pkl"))
        total_scores_ua = checkpoint_data["total_scores_ua"]
        total_scores_wa = checkpoint_data["total_scores_wa"]
        total_scores_f1 = checkpoint_data["total_scores_f1"]
        total_scores_loss_wa = checkpoint_data["total_scores_loss_wa"]
        total_scores_loss_f1 = checkpoint_data["total_scores_loss_f1"]

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
                logger = util.make_logger(os.path.join(
                    exp_dir, f"train_{seed}_fold_{exp_fold}.log"),
                    config)

                MODEL_PATH = f"fold_{exp_fold}_seed_{seed}" \
                             f"_{config.FEATURES_TO_USE}.pth"
                MODEL_PATH = os.path.join(exp_dir, MODEL_PATH)

                if config.DATASET == "cmu":
                    from data_processes.data_processing_cmu import \
                        data_preprocessing
                elif config.DATASET == "ravdess":
                    from data_processes.data_processing_ravdess import \
                        data_preprocessing
                elif config.DATASET == "iemocap":
                    from data_processes.data_processing_iemocap import \
                        data_preprocessing

                feat_dict = data_preprocessing(config, logger, features_exist,
                                               exp_fold)
                features_exist = True

                _ = train_model(feat_dict, logger)
                checkpoint = False
                handlers = logger.handlers[:]
                for handler in handlers:
                    handler.close()
                    logger.removeHandler(handler)

            with torch.no_grad():
                best_scores = util.get_fold_score(config=config,
                                                  seed=seed,
                                                  model_path=MODEL_PATH,
                                                  exp_dir=exp_dir)
            total_scores_ua, total_scores_wa, total_scores_f1, \
            total_scores_loss_wa, total_scores_loss_f1 = util.append_scores(
                config, total_scores_ua, total_scores_wa,
                total_scores_f1, total_scores_loss_wa,
                total_scores_loss_f1, best_scores)

        util.print_res(config, total_scores_ua, total_scores_wa,
                       total_scores_f1, total_scores_loss_wa,
                       total_scores_loss_f1)

    else:
        if config.RUN_INFERENCE == "test" and len(config.DATA_SPLIT) != 3:
            sys.exit("RUN_TEST selected in config but no test features "
                     "exist, try selecting the correct data dir or "
                     "re-creating the dataset with a split for the "
                     "training, validation, and test set in DATA_SPLIT")

        logger = util.make_logger(os.path.join(
            config.EXP_DIR,
            f"{config.RUN_INFERENCE}_skipTrain.log"),
            config)
        with torch.no_grad():
            run_test(config.RUN_INFERENCE)
