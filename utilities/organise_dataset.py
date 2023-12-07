import os
import glob

from configs import config_iemocap


def load_text_file(file):
    with open(file) as f:
        output = f.readlines()
    return output


def run_organiser():
    """
    Reformat the IEMOCAP dataset file names to fit Ses01F_impro01_01_F000 -
    where Ses01F is Session 1 where Female is wearing mocap, impro01 is
    improvised session 1, 01 is the neutral emotion according to the
    dictionary below, and F000 is the female speaking the 0th utterance

    DATABASE_EMO_IDX = {
    '01': 'neu',  # neutral
    '02': 'fru',  # frustration
    '03': 'hap',  # happiness
    '04': 'exc',  # excited
    '05': 'sad',  # sadness
    '06': 'ang',  # angry
    '07': 'fea',  # fear
    '08': 'sur',  # surprise
    '09': 'dis',  # disgust
    '10': 'oth',  # other
    '11': 'xxx'   # no agreement from annotators}

    Returns: None

    """
    for i, current_file in enumerate(text_files):
        lines = load_text_file(current_file)
        for l in lines:
            l_split = l.split("\t")
            for j, data in enumerate(l_split):
                if "Ses" in data:
                    file_info = l_split[j].split("_")
                    session = file_info[0]
                    session_num = int(session[3:-1])
                    utterance = file_info[-1]
                    folder_name = data.replace(f"_{utterance}", "")
                    file_name = data + ".wav"
                    emo = l_split[j + 1]
                    emo_int = config_iemocap.DATABASE_EMO_IDX[emo]
                    current_wav_file = os.path.join(
                        dataset_path,
                        f"Session{str(session_num)}",
                        "sentences/wav",
                        folder_name,
                        file_name)
                    updated_wav_file = current_wav_file.replace(
                        f"{utterance}", f"{emo_int}_{utterance}")
                    assert os.path.exists(current_wav_file)
                    os.rename(current_wav_file, updated_wav_file)


if __name__ == "__main__":
    dataset_path = os.path.join(config_iemocap.DATASET_PATH,
                                "IEMOCAP_full_release")
    emotion_labels_text = "*/dialog/EmoEvaluation/*.txt"

    text_files = glob.glob(os.path.join(dataset_path, emotion_labels_text))

    test_file = os.path.join(
        dataset_path,
        "Session1/sentences/wav/Ses01F_impro01/Ses01F_impro01_F000.wav")
    if os.path.exists(test_file):
        run_organiser()
