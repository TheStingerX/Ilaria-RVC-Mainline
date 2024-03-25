import os
import sys
from dotenv import load_dotenv
import requests
import zipfile
now_dir = os.getcwd()
sys.path.append(now_dir)
load_dotenv()
from infer.modules.vc.modules import VC
from i18n.i18n import I18nAuto
from configs.config import Config
from sklearn.cluster import MiniBatchKMeans
import torch
import numpy as np
import gradio as gr
import faiss
import fairseq
import pathlib
import json
from time import sleep
from subprocess import Popen
from random import shuffle
import warnings
import traceback
import threading
import shutil
import logging
import matplotlib.pyplot as plt
import soundfile as sf
from dotenv import load_dotenv

import edge_tts, asyncio
from infer.modules.vc.ilariatts import tts_order_voice
language_dict = tts_order_voice
ilariavoices = list(language_dict.keys())

now_dir = os.getcwd()
sys.path.append(now_dir)
load_dotenv()

logging.getLogger("numba").setLevel(logging.WARNING)

logger = logging.getLogger(__name__)

tmp = os.path.join(now_dir, "TEMP")
shutil.rmtree(tmp, ignore_errors=True)
shutil.rmtree("%s/runtime/Lib/site-packages/infer_pack" % now_dir, ignore_errors=True)
os.makedirs(tmp, exist_ok=True)
os.makedirs(os.path.join(now_dir, "logs"), exist_ok=True)
os.makedirs(os.path.join(now_dir, "models/pth"), exist_ok=True)
os.environ["TEMP"] = tmp
warnings.filterwarnings("ignore")
torch.manual_seed(114514)

config = Config()
vc = VC(config)

if config.dml:
    def forward_dml(ctx, x, scale):
        ctx.scale = scale
        res = x.clone().detach()
        return res


    fairseq.modules.grad_multiply.GradMultiply.forward = forward_dml
i18n = I18nAuto()
logger.info(i18n)
ngpu = torch.cuda.device_count()
gpu_infos = []
mem = []
if_gpu_ok = False

if torch.cuda.is_available() or ngpu != 0:
    for i in range(ngpu):
        gpu_name = torch.cuda.get_device_name(i)
        if any(
                value in gpu_name.upper()
                for value in [
                    "10",
                    "16",
                    "20",
                    "30",
                    "40",
                    "A2",
                    "A3",
                    "A4",
                    "P4",
                    "A50",
                    "500",
                    "A60",
                    "70",
                    "80",
                    "90",
                    "M4",
                    "T4",
                    "TITAN",
                ]
        ):
            if_gpu_ok = True
            gpu_infos.append("%s\t%s" % (i, gpu_name))
            mem.append(
                int(
                    torch.cuda.get_device_properties(i).total_memory
                    / 1024
                    / 1024
                    / 1024
                    + 0.4
                )
            )
if if_gpu_ok and len(gpu_infos) > 0:
    gpu_info = "\n".join(gpu_infos)
    default_batch_size = min(mem) // 2
else:
    gpu_info = i18n("Your GPU doesn't work for training")
    default_batch_size = 1
gpus = "-".join([i[0] for i in gpu_infos])


class ToolButton(gr.Button, gr.components.FormComponent):

    def __init__(self, **kwargs):
        super().__init__(variant="tool", **kwargs)

    def get_block_name(self):
        return "button"


weight_root = os.getenv("weight_root")
index_root = os.getenv("index_root")

weight_root = "./models/pth"
index_root = "./models/index"

names = []
for name in os.listdir(weight_root):
    if name.endswith(".pth"):
        names.append(name)
index_paths = []
for root, dirs, files in os.walk(index_root, topdown=False):
    for name in files:
        if name.endswith(".index") and "trained" not in name:
            index_paths.append("%s/%s" % (root, name))

def generate_spectrogram(audio_data, sample_rate, file_name):
    plt.clf()

    plt.specgram(
        audio_data,
        Fs=sample_rate / 1,
        NFFT=4096,
        sides="onesided",
        cmap="Reds_r",
        scale_by_freq=True,
        scale="dB",
        mode="magnitude",
        window=np.hanning(4096),
    )

    plt.title(file_name)
    plt.savefig("spectrogram.png")


def get_audio_info(audio_file):
    audio_data, sample_rate = sf.read(audio_file)

    if len(audio_data.shape) > 1:
        audio_data = np.mean(audio_data, axis=1)

    generate_spectrogram(audio_data, sample_rate, os.path.basename(audio_file))

    audio_info = sf.info(audio_file)
    bit_depth = {"PCM_16": 16, "FLOAT": 32}.get(audio_info.subtype, 0)

    minutes, seconds = divmod(audio_info.duration, 60)
    seconds, milliseconds = divmod(seconds, 1)
    milliseconds *= 1000

    speed_in_kbps = audio_info.samplerate * bit_depth / 1000
    # Create a table with the audio file info
    filename_without_extension, _ = os.path.splitext(os.path.basename(audio_file))

    info_table = f"""
    | Information | Value |
    | :---: | :---: |
    | File Name | {filename_without_extension} |
    | Duration | {int(minutes)} minutes - {int(seconds)} seconds - {int(milliseconds)} milliseconds |
    | Bitrate | {speed_in_kbps} kbp/s |
    | Audio Channels | {audio_info.channels} |
    | Samples per second | {audio_info.samplerate} Hz |
    | Bit per second | {audio_info.samplerate * audio_info.channels * bit_depth} bit/s |
    """

    return info_table, "spectrogram.png"

def change_choices():
    names = []
    for name in os.listdir(weight_root):
        if name.endswith(".pth"):
            names.append(name)
    index_paths = []
    for root, dirs, files in os.walk(index_root, topdown=False):
        for name in files:
            if name.endswith(".index") and "trained" not in name:
                index_paths.append("%s/%s" % (root, name))
    return {"choices": sorted(names), "__type__": "update"}, {
        "choices": sorted(index_paths),
        "__type__": "update",
    }


# Define the tts_and_convert function
def tts_and_convert(ttsvoice, text, spk_item, vc_transform, f0_file, f0method, file_index1, file_index2, index_rate, filter_radius, resample_sr, rms_mix_rate, protect):

    # Perform TTS (we only need 1 function)
    vo=language_dict[ttsvoice]
    asyncio.run(edge_tts.Communicate(text, vo).save("./TEMP/temp_ilariatts.mp3"))
    aud_path = './TEMP/temp_ilariatts.mp3'

    # Update output Textbox
    vc_output1.update("Text converted successfully!")

    #Calls vc similar to any other inference.
    #This is why we needed all the other shit in our call, otherwise we couldn't infer.
    return vc.vc_single(spk_item, aud_path, vc_transform, f0_file, f0method, file_index1, file_index2, index_rate, filter_radius, resample_sr, rms_mix_rate, protect)


def import_files(file):
    if file is not None:
        file_name = file.name
        if file_name.endswith('.zip'):
            with zipfile.ZipFile(file.name, 'r') as zip_ref:
                # Create a temporary directory to extract files
                temp_dir = './TEMP'
                zip_ref.extractall(temp_dir)
                # Move .pth and .index files to their respective directories
                for root, dirs, files in os.walk(temp_dir):
                    for file in files:
                        if file.endswith('.pth'):
                            destination = './models/pth/' + file
                            if not os.path.exists(destination):
                                shutil.move(os.path.join(root, file), destination)
                            else:
                                print(f"File {destination} already exists. Skipping.")
                        elif file.endswith('.index'):
                            destination = './models/index/' + file
                            if not os.path.exists(destination):
                                shutil.move(os.path.join(root, file), destination)
                            else:
                                print(f"File {destination} already exists. Skipping.")
                # Remove the temporary directory
                shutil.rmtree(temp_dir)
            return "Zip file has been successfully extracted."
        elif file_name.endswith('.pth'):
            destination = './models/pth/' + os.path.basename(file.name)
            if not os.path.exists(destination):
                os.rename(file.name, destination)
            else:
                print(f"File {destination} already exists. Skipping.")
            return "PTH file has been successfully imported."
        elif file_name.endswith('.index'):
            destination = './models/index/' + os.path.basename(file.name)
            if not os.path.exists(destination):
                os.rename(file.name, destination)
            else:
                print(f"File {destination} already exists. Skipping.")
            return "Index file has been successfully imported."
        else:
            return "Unsupported file type."
    else:
        return "No file has been uploaded."

def import_button_click(file):
    return import_files(file)

      
def clean():
    return {"value": "", "__type__": "update"}


sr_dict = {
    "32k": 32000,
    "40k": 40000,
    "48k": 48000,
}

def if_done(done, p):
    while 1:
        if p.poll() is None:
            sleep(0.5)
        else:
            break
    done[0] = True


def download_from_url(url, model):
    if url == '':
        return "URL cannot be left empty."
    if model == '':
        return "You need to name your model. For example: Ilaria"

    url = url.strip()
    zip_dirs = ["zips", "unzips"]
    for directory in zip_dirs:
        if os.path.exists(directory):
            shutil.rmtree(directory)

    os.makedirs("zips", exist_ok=True)
    os.makedirs("unzips", exist_ok=True)

    zipfile = model + '.zip'
    zipfile_path = './zips/' + zipfile

    try:
        if "drive.google.com" in url:
            subprocess.run(["gdown", url, "--fuzzy", "-O", zipfile_path])
        elif "mega.nz" in url:
            m = Mega()
            m.download_url(url, './zips')
        else:
            response = requests.get(url)
            response.raise_for_status()  # Raise an exception for HTTP errors
            with open(zipfile_path, 'wb') as file:
                file.write(response.content)

        shutil.unpack_archive(zipfile_path, "./unzips", 'zip')

        for root, dirs, files in os.walk('./unzips'):
            for file in files:
                file_path = os.path.join(root, file)
                if file.endswith(".index"):
                    os.makedirs(f'./models/index', exist_ok=True)
                    shutil.copy2(file_path, f'./models/index/{model}.index')
                elif "G_" not in file and "D_" not in file and file.endswith(".pth"):
                    os.makedirs(f'./models/pth', exist_ok=True)
                    shutil.copy(file_path, f'./models/pth/{model}.pth')

        shutil.rmtree("zips")
        shutil.rmtree("unzips")
        return "Model downloaded, you can go back to the inference page!"

    except subprocess.CalledProcessError as e:
        return f"ERROR - Download failed (gdown): {str(e)}"
    except requests.exceptions.RequestException as e:
        return f"ERROR - Download failed (requests): {str(e)}"
    except Exception as e:
        return f"ERROR - The test failed: {str(e)}"


def if_done_multi(done, ps):
    while 1:
        flag = 1
        for p in ps:
            if p.poll() is None:
                flag = 0
                sleep(0.5)
                break
        if flag == 1:
            break
    done[0] = True


def preprocess_dataset(trainset_dir, exp_dir, sr, n_p):
    sr = sr_dict[sr]
    os.makedirs("%s/logs/%s" % (now_dir, exp_dir), exist_ok=True)
    f = open("%s/logs/%s/preprocess.log" % (now_dir, exp_dir), "w")
    f.close()
    per = 3.0 if config.is_half else 3.7
    cmd = '"%s" infer/modules/train/preprocess.py "%s" %s %s "%s/logs/%s" %s %.1f' % (
        config.python_cmd,
        trainset_dir,
        sr,
        n_p,
        now_dir,
        exp_dir,
        config.noparallel,
        per,
    )
    logger.info(cmd)
    p = Popen(cmd, shell=True)
    done = [False]
    threading.Thread(
        target=if_done,
        args=(
            done,
            p,
        ),
    ).start()
    while 1:
        with open("%s/logs/%s/preprocess.log" % (now_dir, exp_dir), "r") as f:
            yield f.read()
        sleep(1)
        if done[0]:
            break
    with open("%s/logs/%s/preprocess.log" % (now_dir, exp_dir), "r") as f:
        log = f.read()
    logger.info(log)
    yield log


def extract_f0_feature(gpus, n_p, f0method, if_f0, exp_dir, version19, gpus_rmvpe):
    gpus = gpus.split("-")
    os.makedirs("%s/logs/%s" % (now_dir, exp_dir), exist_ok=True)
    f = open("%s/logs/%s/extract_f0_feature.log" % (now_dir, exp_dir), "w")
    f.close()
    if if_f0:
        if f0method != "rmvpe_gpu":
            cmd = (
                    '"%s" infer/modules/train/extract/extract_f0_print.py "%s/logs/%s" %s %s'
                    % (
                        config.python_cmd,
                        now_dir,
                        exp_dir,
                        n_p,
                        f0method,
                    )
            )
            logger.info(cmd)
            p = Popen(
                cmd, shell=True, cwd=now_dir
            )
            done = [False]
            threading.Thread(
                target=if_done,
                args=(
                    done,
                    p,
                ),
            ).start()
        else:
            if gpus_rmvpe != "-":
                gpus_rmvpe = gpus_rmvpe.split("-")
                leng = len(gpus_rmvpe)
                ps = []
                for idx, n_g in enumerate(gpus_rmvpe):
                    cmd = (
                            '"%s" infer/modules/train/extract/extract_f0_rmvpe.py %s %s %s "%s/logs/%s" %s '
                            % (
                                config.python_cmd,
                                leng,
                                idx,
                                n_g,
                                now_dir,
                                exp_dir,
                                config.is_half,
                            )
                    )
                    logger.info(cmd)
                    p = Popen(
                        cmd, shell=True, cwd=now_dir
                    )
                    ps.append(p)
                done = [False]
                threading.Thread(
                    target=if_done_multi,  #
                    args=(
                        done,
                        ps,
                    ),
                ).start()
            else:
                cmd = (
                        config.python_cmd
                        + ' infer/modules/train/extract/extract_f0_rmvpe_dml.py "%s/logs/%s" '
                        % (
                            now_dir,
                            exp_dir,
                        )
                )
                logger.info(cmd)
                p = Popen(
                    cmd, shell=True, cwd=now_dir
                )
                p.wait()
                done = [True]
        while 1:
            with open(
                    "%s/logs/%s/extract_f0_feature.log" % (now_dir, exp_dir), "r"
            ) as f:
                yield f.read()
            sleep(1)
            if done[0]:
                break
        with open("%s/logs/%s/extract_f0_feature.log" % (now_dir, exp_dir), "r") as f:
            log = f.read()
        logger.info(log)
        yield log

    leng = len(gpus)
    ps = []
    for idx, n_g in enumerate(gpus):
        cmd = (
                '"%s" infer/modules/train/extract_feature_print.py %s %s %s %s "%s/logs/%s" %s'
                % (
                    config.python_cmd,
                    config.device,
                    leng,
                    idx,
                    n_g,
                    now_dir,
                    exp_dir,
                    version19,
                )
        )
        logger.info(cmd)
        p = Popen(
            cmd, shell=True, cwd=now_dir
        )
        ps.append(p)
    done = [False]
    threading.Thread(
        target=if_done_multi,
        args=(
            done,
            ps,
        ),
    ).start()
    while 1:
        with open("%s/logs/%s/extract_f0_feature.log" % (now_dir, exp_dir), "r") as f:
            yield f.read()
        sleep(1)
        if done[0]:
            break
    with open("%s/logs/%s/extract_f0_feature.log" % (now_dir, exp_dir), "r") as f:
        log = f.read()
    logger.info(log)
    yield log


def get_pretrained_models(path_str, f0_str, sr2):
    if_pretrained_generator_exist = os.access(
        "assets/pretrained%s/%sG%s.pth" % (path_str, f0_str, sr2), os.F_OK
    )
    if_pretrained_discriminator_exist = os.access(
        "assets/pretrained%s/%sD%s.pth" % (path_str, f0_str, sr2), os.F_OK
    )
    if not if_pretrained_generator_exist:
        logger.warning(
            "assets/pretrained%s/%sG%s.pth not exist, will not use pretrained model",
            path_str,
            f0_str,
            sr2,
        )
    if not if_pretrained_discriminator_exist:
        logger.warning(
            "assets/pretrained%s/%sD%s.pth not exist, will not use pretrained model",
            path_str,
            f0_str,
            sr2,
        )
    return (
        "assets/pretrained%s/%sG%s.pth" % (path_str, f0_str, sr2)
        if if_pretrained_generator_exist
        else "",
        "assets/pretrained%s/%sD%s.pth" % (path_str, f0_str, sr2)
        if if_pretrained_discriminator_exist
        else "",
    )

def change_sr2(sr2, if_f0_3, version19):
    path_str = "" if version19 == "v1" else "_v2"
    f0_str = "f0" if if_f0_3 else ""
    return get_pretrained_models(path_str, f0_str, sr2)


def change_version19(sr2, if_f0_3, version19):
    path_str = "" if version19 == "v1" else "_v2"
    if sr2 == "32k" and version19 == "v1":
        sr2 = "40k"
    to_return_sr2 = (
        {"choices": ["40k", "48k"], "__type__": "update", "value": sr2}
        if version19 == "v1"
        else {"choices": ["32k", "40k", "48k"], "__type__": "update", "value": sr2}
    )
    f0_str = "f0" if if_f0_3 else ""
    return (
        *get_pretrained_models(path_str, f0_str, sr2),
        to_return_sr2,
    )


def change_f0(if_f0_3, sr2, version19):
    path_str = "" if version19 == "v1" else "_v2"
    return (
        {"visible": if_f0_3, "__type__": "update"},
        {"visible": if_f0_3, "__type__": "update"},
        *get_pretrained_models(path_str, "f0" if if_f0_3 is True else "", sr2),
    )


def click_train(
        exp_dir1,
        sr2,
        if_f0_3,
        spk_id5,
        save_epoch10,
        total_epoch11,
        batch_size12,
        if_save_latest13,
        pretrained_G14,
        pretrained_D15,
        gpus16,
        if_cache_gpu17,
        if_save_every_weights18,
        version19,
):
    global f0_dir, f0nsf_dir
    exp_dir = "%s/logs/%s" % (now_dir, exp_dir1)
    os.makedirs(exp_dir, exist_ok=True)
    gt_wavs_dir = "%s/0_gt_wavs" % exp_dir
    feature_dir = (
        "%s/3_feature256" % exp_dir
        if version19 == "v1"
        else "%s/3_feature768" % exp_dir
    )
    if if_f0_3:
        f0_dir = "%s/2a_f0" % exp_dir
        f0nsf_dir = "%s/2b-f0nsf" % exp_dir
        names = (
                set([name.split(".")[0] for name in os.listdir(gt_wavs_dir)])
                & set([name.split(".")[0] for name in os.listdir(feature_dir)])
                & set([name.split(".")[0] for name in os.listdir(f0_dir)])
                & set([name.split(".")[0] for name in os.listdir(f0nsf_dir)])
        )
    else:
        names = set([name.split(".")[0] for name in os.listdir(gt_wavs_dir)]) & set(
            [name.split(".")[0] for name in os.listdir(feature_dir)]
        )
    opt = []
    for name in names:
        if if_f0_3:
            opt.append(
                "%s/%s.wav|%s/%s.npy|%s/%s.wav.npy|%s/%s.wav.npy|%s"
                % (
                    gt_wavs_dir.replace("\\", "\\\\"),
                    name,
                    feature_dir.replace("\\", "\\\\"),
                    name,
                    f0_dir.replace("\\", "\\\\"),
                    name,
                    f0nsf_dir.replace("\\", "\\\\"),
                    name,
                    spk_id5,
                )
            )
        else:
            opt.append(
                "%s/%s.wav|%s/%s.npy|%s"
                % (
                    gt_wavs_dir.replace("\\", "\\\\"),
                    name,
                    feature_dir.replace("\\", "\\\\"),
                    name,
                    spk_id5,
                )
            )
    fea_dim = 256 if version19 == "v1" else 768
    if if_f0_3:
        for _ in range(2):
            opt.append(
                "%s/logs/mute/0_gt_wavs/mute%s.wav|%s/logs/mute/3_feature%s/mute.npy|%s/logs/mute/2a_f0/mute.wav.npy"
                "|%s/logs/mute/2b-f0nsf/mute.wav.npy|%s"
                % (now_dir, sr2, now_dir, fea_dim, now_dir, now_dir, spk_id5)
            )
    else:
        for _ in range(2):
            opt.append(
                "%s/logs/mute/0_gt_wavs/mute%s.wav|%s/logs/mute/3_feature%s/mute.npy|%s"
                % (now_dir, sr2, now_dir, fea_dim, spk_id5)
            )
    shuffle(opt)
    with open("%s/filelist.txt" % exp_dir, "w") as f:
        f.write("\n".join(opt))
    logger.debug("Write filelist done")
    logger.info("Use gpus: %s", str(gpus16))
    if pretrained_G14 == "":
        logger.info("No pretrained Generator")
    if pretrained_D15 == "":
        logger.info("No pretrained Discriminator")
    if version19 == "v1" or sr2 == "40k":
        config_path = "v1/%s.json" % sr2
    else:
        config_path = "v2/%s.json" % sr2
    config_save_path = os.path.join(exp_dir, "config.json")
    if not pathlib.Path(config_save_path).exists():
        with open(config_save_path, "w", encoding="utf-8") as f:
            json.dump(
                config.json_config[config_path],
                f,
                ensure_ascii=False,
                indent=4,
                sort_keys=True,
            )
            f.write("\n")
    if gpus16:
        cmd = (
                '"%s" infer/modules/train/train.py -e "%s" -sr %s -f0 %s -bs %s -g %s -te %s -se %s %s %s -l %s -c %s '
                "-sw %s -v %s"
                % (
                    config.python_cmd,
                    exp_dir1,
                    sr2,
                    1 if if_f0_3 else 0,
                    batch_size12,
                    gpus16,
                    total_epoch11,
                    save_epoch10,
                    "-pg %s" % pretrained_G14 if pretrained_G14 != "" else "",
                    "-pd %s" % pretrained_D15 if pretrained_D15 != "" else "",
                    1 if if_save_latest13 == i18n("是") else 0,
                    1 if if_cache_gpu17 == i18n("是") else 0,
                    1 if if_save_every_weights18 == i18n("是") else 0,
                    version19,
                )
        )
    else:
        cmd = (
                '"%s" infer/modules/train/train.py -e "%s" -sr %s -f0 %s -bs %s -te %s -se %s %s %s -l %s -c %s -sw '
                "%s -v %s"
                % (
                    config.python_cmd,
                    exp_dir1,
                    sr2,
                    1 if if_f0_3 else 0,
                    batch_size12,
                    total_epoch11,
                    save_epoch10,
                    "-pg %s" % pretrained_G14 if pretrained_G14 != "" else "",
                    "-pd %s" % pretrained_D15 if pretrained_D15 != "" else "",
                    1 if if_save_latest13 == i18n("是") else 0,
                    1 if if_cache_gpu17 == i18n("是") else 0,
                    1 if if_save_every_weights18 == i18n("是") else 0,
                    version19,
                )
        )
    logger.info(cmd)
    p = Popen(cmd, shell=True, cwd=now_dir)
    p.wait()
    return "You can view console or train.log"


def train_index(exp_dir1, version19):
    exp_dir = "logs/%s" % exp_dir1
    os.makedirs(exp_dir, exist_ok=True)
    feature_dir = (
        "%s/3_feature256" % exp_dir
        if version19 == "v1"
        else "%s/3_feature768" % exp_dir
    )
    if not os.path.exists(feature_dir):
        return "Please perform Feature Extraction First!"
    listdir_res = list(os.listdir(feature_dir))
    if len(listdir_res) == 0:
        return "Please perform Feature Extraction First！"
    infos = []
    npys = []
    for name in sorted(listdir_res):
        phone = np.load("%s/%s" % (feature_dir, name))
        npys.append(phone)
    big_npy = np.concatenate(npys, 0)
    big_npy_idx = np.arange(big_npy.shape[0])
    np.random.shuffle(big_npy_idx)
    big_npy = big_npy[big_npy_idx]
    if big_npy.shape[0] > 2e5:
        infos.append("Trying doing kmeans %s shape to 10k centers." % big_npy.shape[0])
        yield "\n".join(infos)
        try:
            big_npy = (
                MiniBatchKMeans(
                    n_clusters=10000,
                    verbose=True,
                    batch_size=256 * config.n_cpu,
                    compute_labels=False,
                    init="random",
                )
                .fit(big_npy)
                .cluster_centers_
            )
        except:
            info = traceback.format_exc()
            logger.info(info)
            infos.append(info)
            yield "\n".join(infos)

    np.save("%s/total_fea.npy" % exp_dir, big_npy)
    n_ivf = min(int(16 * np.sqrt(big_npy.shape[0])), big_npy.shape[0] // 39)
    infos.append("%s,%s" % (big_npy.shape, n_ivf))
    yield "\n".join(infos)
    index = faiss.index_factory(256 if version19 == "v1" else 768, "IVF%s,Flat" % n_ivf)
    infos.append("training")
    yield "\n".join(infos)
    index_ivf = faiss.extract_index_ivf(index)  #
    index_ivf.nprobe = 1
    index.train(big_npy)
    faiss.write_index(
        index,
        "%s/trained_IVF%s_Flat_nprobe_%s_%s_%s.index"
        % (exp_dir, n_ivf, index_ivf.nprobe, exp_dir1, version19),
    )

    infos.append("adding")
    yield "\n".join(infos)
    batch_size_add = 8192
    for i in range(0, big_npy.shape[0], batch_size_add):
        index.add(big_npy[i: i + batch_size_add])
    faiss.write_index(
        index,
        "%s/added_IVF%s_Flat_nprobe_%s_%s_%s.index"
        % (exp_dir, n_ivf, index_ivf.nprobe, exp_dir1, version19),
    )
    infos.append(
        "Success，added_IVF%s_Flat_nprobe_%s_%s_%s.index"
        % (n_ivf, index_ivf.nprobe, exp_dir1, version19)
    )
    yield "\n".join(infos)


F0GPUVisible = config.dml is False


def change_f0_method(f0method8):
    if f0method8 == "rmvpe_gpu":
        visible = F0GPUVisible
    else:
        visible = False
    return {"visible": visible, "__type__": "update"}

vc_output1 = gr.Textbox(label=i18n("Console"))
vc_output2 = gr.Audio(label=i18n("Audio output"))

with gr.Blocks(title="Ilaria RVC 💖") as app:
    gr.Markdown("<h1>  Ilaria RVC 💖   </h1>")
    gr.Markdown(value=i18n("Made with 💖 by Ilaria | Support her on [Ko-Fi](https://ko-fi.com/ilariaowo)"))
    with gr.Tabs():
        with gr.TabItem(i18n("Inference")):
            with gr.Row():
                sid0= gr.Dropdown(label=i18n("Voice"), choices=sorted(names))
                sid1= sid0
                
                with gr.Column():
                    refresh_button = gr.Button(i18n("Refresh"), variant="primary")
                    clean_button = gr.Button(i18n("Unload Voice from VRAM"), variant="primary")
                spk_item = gr.Slider(
                    minimum=0,
                    maximum=2333,
                    step=1,
                    label=i18n("Speaker ID (Auto-Detected)"),
                    value=0,
                    visible=True,
                    interactive=False,
                )
                clean_button.click(
                    fn=clean, inputs=[], outputs=[sid0], api_name="infer_clean"
                )
            with gr.TabItem(i18n("Inference")):
                with gr.Group():
                    with gr.Row():
                        with gr.Column():
                            with gr.Accordion('Audio input', open=True):

                                
                                input_audio0 = gr.Audio(
                                    label=i18n("Upload Audio file"),
                                    type="filepath",
                                )
                                record_button = gr.Audio(source="microphone", label="Or you can use your microphone!",
                                                         type="filepath")
                                record_button.change(
                                    fn=lambda x: x,
                                    inputs=[record_button],
                                    outputs=[input_audio0],
                                )
                                file_index1 = gr.Textbox(
                                    label=i18n("Path of index"),
                                    placeholder="%userprofile%\\Desktop\\models\\model_example.index",
                                    interactive=True,
                                    visible=False,
                                )
                                file_index2 = gr.Textbox(
                                    label=i18n("Auto-detect index path"),
                                    choices=sorted(index_paths),
                                    interactive=True,
                                    visible=False,
                                )
                        with gr.Column():
                            
                            vc_transform0 = gr.inputs.Slider(
                                label=i18n(
                                    "Pitch: 0 from man to man (or woman to woman); 12 from man to woman and -12 from woman to man."),
                                minimum=-12,
                                maximum=12,
                                default=0,
                                step=1,
                            )
                                    
                            with gr.Accordion('Advanced Settings', open=False, visible=False):
                                with gr.Column():
                                    f0method0 = gr.Radio(
                                        label=i18n("Pitch Extraction, rmvpe is best"),
                                        choices=["harvest", "crepe", "rmvpe"]
                                        if config.dml is False
                                        else ["harvest", "rmvpe"],
                                        value="rmvpe",
                                        interactive=True,
                                    )
                                    resample_sr0 = gr.Slider(
                                        minimum=0,
                                        maximum=48000,
                                        label=i18n("Resampling, 0=none"),
                                        value=0,
                                        step=1,
                                        interactive=True,
                                    )
                                    rms_mix_rate0 = gr.Slider(
                                        minimum=0,
                                        maximum=1,
                                        label=i18n("0=Input source volume, 1=Normalized Output"),
                                        value=0.25,
                                        interactive=True,
                                    )
                                    protect0 = gr.Slider(
                                        minimum=0,
                                        maximum=0.5,
                                        label=i18n(
                                            "Protect clear consonants and breathing sounds, preventing electro-acoustic tearing and other artifacts, 0.5 does not open"),
                                        value=0.33,
                                        step=0.01,
                                        interactive=True,
                                    )
                                    filter_radius0 = gr.Slider(
                                        minimum=0,
                                        maximum=7,
                                        label=i18n(">=3 apply median filter to the harvested pitch results"),
                                        value=3,
                                        step=1,
                                        interactive=True,
                                    )
                                    index_rate1 = gr.Slider(
                                        minimum=0,
                                        maximum=1,
                                        label=i18n("Index Ratio"),
                                        value=0.40,
                                        interactive=True,
                                    )
                                    f0_file = gr.File(
                                        label=i18n("F0 curve file [optional]"),
                                        visible=False,
                                    )

                                    refresh_button.click(
                                        fn=change_choices,
                                        inputs=[],
                                        outputs=[sid0, file_index2],
                                        api_name="infer_refresh",
                                    )
                                    file_index1 = gr.Textbox(
                                        label=i18n("Path of index"),
                                        placeholder="%userprofile%\\Desktop\\models\\model_example.index",
                                        interactive=True,
                                    )
                                    file_index2 = gr.Dropdown(
                                        label=i18n("Auto-detect index path"),
                                        choices=sorted(index_paths),
                                        interactive=True,
                                    )
                                    
                            with gr.Accordion('IlariaTTS', open=True):
                                with gr.Column():
                                    ilariaid=gr.Dropdown(label="Voice:", choices=ilariavoices, interactive=True, value="English-Jenny (Female)")
                                    ilariatext = gr.Textbox(label="Input your Text", interactive=True, value="This is a test.")   
                                    ilariatts_button = gr.Button(value="Speak and Convert")
                                    ilariatts_button.click(tts_and_convert,
                                                           [ilariaid,
                                                            ilariatext,
                                                            spk_item,
                                                            vc_transform0,
                                                            f0_file,
                                                            f0method0,
                                                            file_index1,
                                                            file_index2,
                                                            index_rate1,
                                                            filter_radius0,
                                                            resample_sr0,
                                                            rms_mix_rate0,
                                                            protect0]
                                                           , [vc_output1, vc_output2])
                            
                                      #Otherwise everything break, to be optimized
                            with gr.Accordion('Advanced Settings', open=False, visible=True):
                                with gr.Column():
                                    f0method0 = gr.Radio(
                                        label=i18n("Pitch Extraction, rmvpe is best"),
                                        choices=["harvest", "crepe", "rmvpe"]
                                        if config.dml is False
                                        else ["harvest", "rmvpe"],
                                        value="rmvpe",
                                        interactive=True,
                                    )
                                    resample_sr0 = gr.Slider(
                                        minimum=0,
                                        maximum=48000,
                                        label=i18n("Resampling, 0=none"),
                                        value=0,
                                        step=1,
                                        interactive=True,
                                    )
                                    rms_mix_rate0 = gr.Slider(
                                        minimum=0,
                                        maximum=1,
                                        label=i18n("0=Input source volume, 1=Normalized Output"),
                                        value=0.25,
                                        interactive=True,
                                    )
                                    protect0 = gr.Slider(
                                        minimum=0,
                                        maximum=0.5,
                                        label=i18n(
                                            "Protect clear consonants and breathing sounds, preventing electro-acoustic tearing and other artifacts, 0.5 does not open"),
                                        value=0.33,
                                        step=0.01,
                                        interactive=True,
                                    )
                                    filter_radius0 = gr.Slider(
                                        minimum=0,
                                        maximum=7,
                                        label=i18n(">=3 apply median filter to the harvested pitch results"),
                                        value=3,
                                        step=1,
                                        interactive=True,
                                    )
                                    index_rate1 = gr.Slider(
                                        minimum=0,
                                        maximum=1,
                                        label=i18n("Index Ratio"),
                                        value=0.40,
                                        interactive=True,
                                    )
                                    f0_file = gr.File(
                                        label=i18n("F0 curve file [optional]"),
                                        visible=False,
                                    )

                                    refresh_button.click(
                                        fn=change_choices,
                                        inputs=[],
                                        outputs=[sid0, file_index2],
                                        api_name="infer_refresh",
                                    )
                                    file_index1 = gr.Textbox(
                                        label=i18n("Path of index"),
                                        placeholder="%userprofile%\\Desktop\\models\\model_example.index",
                                        interactive=True,
                                    )
                                    file_index2 = gr.Dropdown(
                                        label=i18n("Auto-detect index path"),
                                        choices=sorted(index_paths),
                                        interactive=True,
                                    )

                with gr.Group():
                    with gr.Column():
                        but0 = gr.Button(i18n("Convert"), variant="primary")
                        with gr.Row():
                            vc_output1.render()
                            vc_output2.render()

                        but0.click(
                            vc.vc_single,
                            [
                                spk_item,
                                input_audio0,
                                vc_transform0,
                                f0_file,
                                f0method0,
                                file_index1,
                                file_index2,
                                # file_big_npy1,
                                index_rate1,
                                filter_radius0,
                                resample_sr0,
                                rms_mix_rate0,
                                protect0,
                            ],
                            [vc_output1, vc_output2],
                            api_name="infer_convert",
                        )
            with gr.TabItem("Download Voice Models"):
                with gr.Row():
                    url = gr.Textbox(label="Huggingface Link:")
                with gr.Row():
                    model = gr.Textbox(label="Name of the model (without spaces):")
                    download_button = gr.Button("Download")
                with gr.Row():
                    status_bar = gr.Textbox(label="Download Status")
                download_button.click(fn=download_from_url, inputs=[url, model], outputs=[status_bar])

            with gr.TabItem("Import Models"):
             file_upload = gr.File(label="Upload a .zip file containing a .pth and .index file")
             import_button = gr.Button("Import")
             import_status = gr.Textbox(label="Import Status")
             import_button.click(fn=import_button_click, inputs=file_upload, outputs=import_status)

            with gr.TabItem(i18n("Batch Inference")):
                gr.Markdown(
                    value=i18n("Batch Conversion")
                )
                
                with gr.Row():
                    with gr.Column():
                        vc_transform1 = gr.Number(
                            label=i18n("Pitch: 0 from man to man (or woman to woman); 12 from man to woman and -12 from woman to man."),
                            value=0
                        )
                        opt_input = gr.Textbox(label=i18n("Output"), value="InferOutput")
                        file_index3 = gr.Textbox(
                            label=i18n("Path to index"),
                            value="",
                            interactive=True,
                        )
                        file_index4 = gr.Dropdown(
                            label=i18n("Auto-detect index path"),
                            choices=sorted(index_paths),
                            interactive=True,
                        )
                        f0method1 = gr.Radio(
                            label=i18n("Pitch Extraction, rmvpe is best"),
                            choices=["harvest", "crepe", "rmvpe"]
                            if config.dml is False
                            else ["harvest", "rmvpe"],
                            value="rmvpe",
                            interactive=True,
                        )
                        format1 = gr.Radio(
                            label=i18n("Export Format"),
                            choices=["flac", "wav", "mp3", "m4a"],
                            value="flac",
                            interactive=True,
                        )

                        refresh_button.click(
                            fn=lambda: change_choices()[1],
                            inputs=[],
                            outputs=file_index4,
                            api_name="infer_refresh_batch",
                        )

                    with gr.Column():
                        resample_sr1 = gr.Slider(
                            minimum=0,
                            maximum=48000,
                            label=i18n("Resampling, 0=none"),
                            value=0,
                            step=1,
                            interactive=True,
                        )
                        rms_mix_rate1 = gr.Slider(
                            minimum=0,
                            maximum=1,
                            label=i18n("0=Input source volume, 1=Normalized Output"),
                            value=0.25,
                            interactive=True,
                        )
                        protect1 = gr.Slider(
                            minimum=0,
                            maximum=0.5,
                            label=i18n(
                                "Protect clear consonants and breathing sounds, preventing electro-acoustic tearing and other artifacts, 0.5 does not open"),
                            value=0.33,
                            step=0.01,
                            interactive=True,
                        )
                        filter_radius1 = gr.Slider(
                            minimum=0,
                            maximum=7,
                            label=i18n(">=3 apply median filter to the harvested pitch results"),
                            value=3,
                            step=1,
                            interactive=True,
                        )
                        index_rate2 = gr.Slider(
                            minimum=0,
                            maximum=1,
                            label=i18n("Index Ratio"),
                            value=0.40,
                            interactive=True,
                        )
                with gr.Row():
                    dir_input = gr.Textbox(
                        label=i18n("Enter the path to the audio folder to be processed"),
                        placeholder="%userprofile%\\Desktop\\covers",
                    )
                    inputs = gr.File(
                        file_count="multiple", label=i18n("Audio files can also be imported in batch")
                    )

                with gr.Row():
                    but1 = gr.Button(i18n("Convert"), variant="primary")
                    vc_output3 = gr.Textbox(label=i18n("Console"))

                    but1.click(
                        vc.vc_multi,
                        [
                            spk_item,
                            dir_input,
                            opt_input,
                            inputs,
                            vc_transform1,
                            f0method1,
                            file_index3,
                            file_index4,
                            # file_big_npy2,
                            index_rate2,
                            filter_radius1,
                            resample_sr1,
                            rms_mix_rate1,
                            protect1,
                            format1,
                        ],
                        [vc_output3],
                        api_name="infer_convert_batch",
                    )
        with gr.TabItem(i18n("Train")):
            gr.Markdown(value=i18n(""))
            with gr.Row():
                exp_dir1 = gr.Textbox(label=i18n("Model Name"), value="test-model")
                sr2 = gr.Radio(
                    label=i18n("Sample Rate"),
                    choices=["32k", "40k", "48k"],
                    value="32k",
                    interactive=True,
                )
                if_f0_3 = gr.Radio(
                    label=i18n("Pitch Guidance"),
                    choices=[True, False],
                    value=True,
                    interactive=True,
                )
                version19 = gr.Radio(
                    label=i18n("Version 2 only here"),
                    choices=["v2"],
                    value="v2",
                    interactive=False,
                    visible=False,
                )
                np7 = gr.Slider(
                    minimum=0,
                    maximum=config.n_cpu,
                    step=1,
                    label=i18n("CPU Threads"),
                    value=int(np.ceil(config.n_cpu / 2.5)),
                    interactive=True,
                )
            with gr.Group():
                gr.Markdown(value=i18n(""))
                with gr.Row():
                    trainset_dir4 = gr.Textbox(
                        label=i18n("Path to Dataset"), value="dataset"
                    )
                    spk_id5 = gr.Slider(
                        minimum=0,
                        maximum=4,
                        step=1,
                        label=i18n("Speaker ID"),
                        value=0,
                        interactive=True,
                    )
                    but1 = gr.Button(i18n("Process Data"), variant="primary")
                    info1 = gr.Textbox(label=i18n("Output"), value="")
                    but1.click(
                        preprocess_dataset,
                        [trainset_dir4, exp_dir1, sr2, np7],
                        [info1],
                        api_name="train_preprocess",
                    )
            with gr.Group():
                gr.Markdown(value=i18n(""))
                with gr.Row():
                    with gr.Column():
                        gpus6 = gr.Textbox(
                            label=i18n("GPU ID (Leave 0 if you have only one GPU, use 0-1 for multiple GPus)"),
                            value=gpus,
                            interactive=True,
                            visible=F0GPUVisible,
                        )
                        gpu_info9 = gr.Textbox(
                            label=i18n("GPU Model"),
                            value=gpu_info,
                            visible=F0GPUVisible,
                        )
                    with gr.Column():
                        f0method8 = gr.Radio(
                            label=i18n("Feature Extraction Method"),
                            choices=["rmvpe", "rmvpe_gpu"],
                            value="rmvpe_gpu",
                            interactive=True,
                        )
                        gpus_rmvpe = gr.Textbox(
                            label=i18n(
                                "rmvpe_gpu will use your GPU instead of the CPU for the feature extraction"
                            ),
                            value="%s-%s" % (gpus, gpus),
                            interactive=True,
                            visible=F0GPUVisible,
                        )
                    but2 = gr.Button(i18n("Feature Extraction"), variant="primary")
                    info2 = gr.Textbox(label=i18n("Output"), value="", max_lines=8)
                    f0method8.change(
                        fn=change_f0_method,
                        inputs=[f0method8],
                        outputs=[gpus_rmvpe],
                    )
                    but2.click(
                        extract_f0_feature,
                        [
                            gpus6,
                            np7,
                            f0method8,
                            if_f0_3,
                            exp_dir1,
                            version19,
                            gpus_rmvpe,
                        ],
                        [info2],
                        api_name="train_extract_f0_feature",
                    )
            with gr.Group():
                gr.Markdown(value=i18n(""))
                with gr.Row():
                    save_epoch10 = gr.Slider(
                        minimum=1,
                        maximum=250,
                        step=1,
                        label=i18n("Save frequency"),
                        value=50,
                        interactive=True,
                    )
                    total_epoch11 = gr.Slider(
                        minimum=2,
                        maximum=10000,
                        step=1,
                        label=i18n("Total Epochs"),
                        value=300,
                        interactive=True,
                    )
                    batch_size12 = gr.Slider(
                        minimum=1,
                        maximum=16,
                        step=1,
                        label=i18n("Batch Size"),
                        value=default_batch_size,
                        interactive=True,
                    )
                    if_save_latest13 = gr.Radio(
                        label=i18n("Save last ckpt as final Model"),
                        choices=[i18n("是"), i18n("否")],
                        value=i18n("是"),
                        interactive=True,
                    )
                    if_cache_gpu17 = gr.Radio(
                        label=i18n("Cache data to GPU (Only for datasets under 8 minutes)"),
                        choices=[i18n("是"), i18n("否")],
                        value=i18n("否"),
                        interactive=True,
                    )
                    if_save_every_weights18 = gr.Radio(
                        label=i18n("Create model with save frequency"),
                        choices=[i18n("是"), i18n("否")],
                        value=i18n("是"),
                        interactive=True,
                    )

                file_dict = {f: os.path.join("assets/pretrained_v2", f) for f in os.listdir("assets/pretrained_v2")}
                file_dict = {k: v for k, v in file_dict.items() if k.endswith(".pth")}
                file_dict_g = {k: v for k, v in file_dict.items() if "G" in k and "f0" in k}
                file_dict_d = {k: v for k, v in file_dict.items() if "D" in k and "f0" in k}

            with gr.Row():
                pretrained_G14 = gr.Dropdown(
                    label=i18n("Pretrained G"),
                    choices=list(file_dict_g.keys()),
                    value=file_dict_g['f0G32k.pth'],
                    interactive=True,
                )
                pretrained_G14_path = gr.State(value=file_dict_g['f0G32k.pth'])

                pretrained_D15 = gr.Dropdown(
                    label=i18n("Pretrained D"),
                    choices=list(file_dict_d.keys()),
                    value=file_dict_d['f0D32k.pth'],
                    interactive=True,
                )
                pretrained_D15_path = gr.State(value=file_dict_d['f0D32k.pth'])

                def update_pretrained_path(choice, state):
                 return os.path.join("assets/pretrained_v2", choice)
                pretrained_G14.change(update_pretrained_path, inputs=[pretrained_G14, pretrained_G14_path], outputs=pretrained_G14_path)
                pretrained_D15.change(update_pretrained_path, inputs=[pretrained_D15, pretrained_D15_path], outputs=pretrained_D15_path)
                sr2.change(
                    change_sr2,
                    [sr2, if_f0_3, version19],
                    [pretrained_G14, pretrained_D15],
                )
                version19.change(
                    change_version19,
                    [sr2, if_f0_3, version19],
                    [pretrained_G14, pretrained_D15, sr2],
                )
                if_f0_3.change(
                    change_f0,
                    [if_f0_3, sr2, version19],
                    [f0method8, gpus_rmvpe, pretrained_G14, pretrained_D15],
                )
                gpus16 = gr.Textbox(
                    label=i18n("Enter cards to be used (Leave 0 if you have only one GPU, use 0-1 for multiple GPus)"),
                    value=gpus,
                    interactive=True,
                )
                but3 = gr.Button(i18n("Train Model"), variant="primary")
                but4 = gr.Button(i18n("Train Index"), variant="primary")
                info3 = gr.Textbox(label=i18n("Output"), value="", max_lines=10)
                but3.click(
                    click_train,
                    [
                        exp_dir1,
                        sr2,
                        if_f0_3,
                        spk_id5,
                        save_epoch10,
                        total_epoch11,
                        batch_size12,
                        if_save_latest13,
                        pretrained_G14_path,
                        pretrained_D15_path,
                        gpus16,
                        if_cache_gpu17,
                        if_save_every_weights18,
                        version19,
                    ],
                    info3,
                    api_name="train_start",
                )
                but4.click(train_index, [exp_dir1, version19], info3)
        
        with gr.TabItem(i18n("Extra")):
                with gr.Accordion('Model Info', open=False):
                    with gr.Column():
                        sid1 = gr.Dropdown(label=i18n("Voice Model"), choices=sorted(names))
                        modelload_out = gr.Textbox(label="Model Metadata")
                        
                with gr.Accordion('Audio Analyser', open=False):
                    with gr.Column():
                        audio_input = gr.Audio(type="filepath")
                        get_info_button = gr.Button(
                            value=i18n("Get information about the audio"), variant="primary"
                        )
                    with gr.Column():
                        with gr.Row():
                            with gr.Column():
                                gr.Markdown(
                                    value=i18n("Information about the audio file"),
                                    visible=True,
                                )
                                output_markdown = gr.Markdown(
                                    value=i18n("Waiting for information..."), visible=True
                                )
                            image_output = gr.Image(type="filepath", interactive=False)

                    get_info_button.click(
                        fn=get_audio_info,
                        inputs=[audio_input],
                        outputs=[output_markdown, image_output],
                    )
                with gr.Accordion('Credits', open=False):
                    gr.Markdown('''
                ## All the amazing people who worked on this!
                
                ### Developers
                
                - **Ilaria**: Founder, Lead Developer
                - **Yui**: Training feature
                - **GDR-**: Inference feature
                - **Poopmaster**: Model downloader, Model importer
                - **kitlemonfoot**: Ilaria TTS implementation
                - **eddycrack864**: UVR5 implementation
                
                ### Beta Tester
                
                - **Charlotte**: Beta Tester
                - **RME**: Beta Tester
                - **Delik**: Beta Tester
                
                ### Pretrains Makers

                - **simplcup**: Ov2Super
                - **mustar22**: RIN_E3
                - **mustar22**: Snowie
                
                ### Other
                
                - **yumereborn**: Ilaira RVC image
                                
                ### **In loving memory of JLabDX** 🕊️
                ''')
                
                sid0.change(
                    fn=vc.get_vc,
                    inputs=[sid0, protect0, protect1],
                    outputs=[spk_item, protect0, protect1, file_index2, file_index4, modelload_out],
                    api_name="infer_change_voice",
                )      
                sid1.change(
                    fn=vc.get_vc,
                    inputs=[sid1, protect0, protect1],
                    outputs=[spk_item, protect0, protect1, file_index2, file_index4, modelload_out],
                    api_name="infer_change_voice",
                )                        
        with gr.TabItem(i18n("")):
            gr.Markdown('''
                ![ilaria](https://i.ytimg.com/vi/5PWqt2Wg-us/maxresdefault.jpg)
            ''')
    if config.iscolab:
        app.queue(concurrency_count=511, max_size=1022).launch(share=True)
    else:
        app.queue(concurrency_count=511, max_size=1022).launch(
            server_name="0.0.0.0",
            inbrowser=not config.noautoopen,
            server_port=config.listen_port,
            quiet=True,
        )
