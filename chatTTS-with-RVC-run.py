#!/usr/bin/env python
# coding: utf-8

# # 脚本介绍
# 将chatTTS的语音通过RVC进行换声
# 并且将中间特征和换声后的Mel频谱图保存下来
# 以便用于训练音色固定的模型

# In[1]:


## 前置依赖
import random
import wave
import numpy as np
import torchaudio
import ChatTTS
from scipy.io.wavfile import write
import librosa

# from zh_normalization import TextNormalizer
import logging
import torch
import os
from IPython.display import Audio

torch._dynamo.config.cache_size_limit = 64
torch._dynamo.config.suppress_errors = True
torch.set_float32_matmul_precision('high')
logging.basicConfig(level=logging.DEBUG)


# ### 加载chaTTS模型

# In[2]:


from modelscope import snapshot_download
model_dir = snapshot_download('mirror013/ChatTTS')

# 加载模型
chat = ChatTTS.Chat()
chat.load_models(
    source="local",
    local_path=model_dir,
    device='cpu',
    compile=False,
)

SEED = 1397
torch.manual_seed(SEED) # 音色种子
# load from local file if exists
if os.path.exists('spk_emb.npy'):
    spk_emb = torch.load('spk_emb.npy',map_location='cpu')
else:
    spk_emb = chat.sample_random_speaker()

params_infer_code = {
    'spk_emb': spk_emb,
    'temperature': 0.1,
    'top_P': 0.7,
    'top_K': 20,
}

params_refine_text = {'prompt': '[oral_0][laugh_0][break_0]'}

text = "接下来,杨叔，借我看一下现场地图。他肯定穿过了前面的那扇门，不可能在这么小的地方晃悠了两小时。" # 该文本仅作测试用途


# ### RVC 依赖函数

# In[3]:


from scipy.io import wavfile
from fairseq import checkpoint_utils
import torchaudio
from lib.audio import load_audio
from lib.infer_pack.models import (
    SynthesizerTrnMs256NSFsid,
    SynthesizerTrnMs256NSFsid_nono,
    SynthesizerTrnMs768NSFsid,
    SynthesizerTrnMs768NSFsid_nono,
)
from vc_infer_pipeline import VC
from multiprocessing import cpu_count
import numpy as np
import torch
import sys
import glob
import argparse
import os
import sys
import pdb
import torch

now_dir = os.getcwd()
sys.path.append(now_dir)

using_cli = False
device = "cuda:0" if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
is_half = False

    
if device == 'mps':
    # 设置环境变量 PYTORCH_ENABLE_MPS_FALLBACK=1
    os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'

# 只在jupyter notebook中运行
from IPython import get_ipython
if get_ipython() is not None:
    get_ipython().run_line_magic('set_env', 'PYTORCH_ENABLE_MPS_FALLBACK=1')
    pass

class Config:
    def __init__(self, device, is_half):
        self.device = device
        self.is_half = is_half
        self.n_cpu = 0
        self.gpu_name = None
        self.gpu_mem = None
        self.x_pad, self.x_query, self.x_center, self.x_max = self.device_config()

    def device_config(self) -> tuple:
        if torch.cuda.is_available() and device != "cpu":
            i_device = int(self.device.split(":")[-1])
            self.gpu_name = torch.cuda.get_device_name(i_device)
            if (
                ("16" in self.gpu_name and "V100" not in self.gpu_name.upper())
                or "P40" in self.gpu_name.upper()
                or "1060" in self.gpu_name
                or "1070" in self.gpu_name
                or "1080" in self.gpu_name
            ):
                print("16系/10系显卡和P40强制单精度")
                self.is_half = False
                for config_file in ["32k.json", "40k.json", "48k.json"]:
                    with open(f"configs/{config_file}", "r") as f:
                        strr = f.read().replace("true", "false")
                    with open(f"configs/{config_file}", "w") as f:
                        f.write(strr)
                with open("trainset_preprocess_pipeline_print.py", "r") as f:
                    strr = f.read().replace("3.7", "3.0")
                with open("trainset_preprocess_pipeline_print.py", "w") as f:
                    f.write(strr)
            else:
                self.gpu_name = None
            self.gpu_mem = int(
                torch.cuda.get_device_properties(i_device).total_memory
                / 1024
                / 1024
                / 1024
                + 0.4
            )
            if self.gpu_mem <= 4:
                with open("trainset_preprocess_pipeline_print.py", "r") as f:
                    strr = f.read().replace("3.7", "3.0")
                with open("trainset_preprocess_pipeline_print.py", "w") as f:
                    f.write(strr)
        elif torch.backends.mps.is_available():
            print("没有发现支持的N卡, 使用MPS进行推理")
            self.device = "mps"
        else:
            print("没有发现支持的N卡, 使用CPU进行推理")
            self.device = "cpu"
            self.is_half = False

        if self.n_cpu == 0:
            self.n_cpu = cpu_count()

        if self.is_half:
            # 6G显存配置
            x_pad = 3
            x_query = 10
            x_center = 60
            x_max = 65
        else:
            # 5G显存配置
            x_pad = 1
            x_query = 6
            x_center = 38
            x_max = 41

        if self.gpu_mem != None and self.gpu_mem <= 4:
            x_pad = 1
            x_query = 5
            x_center = 30
            x_max = 32

        return x_pad, x_query, x_center, x_max


config = Config(device, is_half)
now_dir = os.getcwd()
sys.path.append(now_dir)

hubert_model = None


def load_hubert():
    global hubert_model
    models, _, _ = checkpoint_utils.load_model_ensemble_and_task(
        ["hubert_base.pt"],
        suffix="",
    )
    hubert_model = models[0]
    hubert_model = hubert_model.to(config.device)
    if config.is_half:
        hubert_model = hubert_model.half()
    else:
        hubert_model = hubert_model.float()
    hubert_model.eval()

last_model_path = None
def vc_single(
    sid=0,
    audio=None, # 需要确保是16000采样率
    f0_up_key=0,
    f0_file=None,
    f0_method="rmvpe",
    file_index="",  # .index file
    file_index2="",
    # file_big_npy,
    index_rate=1.0,
    filter_radius=3,
    resample_sr=0,
    rms_mix_rate=0,
    model_path="",
    output_path="",
    protect=0.33,
):
    
    global tgt_sr, net_g, vc, hubert_model, version, last_model_path
    if last_model_path != model_path:
        last_model_path = get_vc(model_path)
    if audio is None:
        raise "You need to upload an audio file"
    if not isinstance(audio,np.ndarray):
        raise "Make sure audio is a numpy array"

    f0_up_key = int(f0_up_key)
    audio_max = np.abs(audio).max() / 0.95

    if audio_max > 1:
        audio /= audio_max
    times = [0, 0, 0]

    if hubert_model == None:
        load_hubert()

    if_f0 = cpt.get("f0", 1)

    file_index = (
        (
            file_index.strip(" ")
            .strip('"')
            .strip("\n")
            .strip('"')
            .strip(" ")
            .replace("trained", "added")
        )
        if file_index != ""
        else file_index2
    )

    audio_opt = vc.pipeline(
        hubert_model,
        net_g,
        sid,
        audio,
        "",
        times,
        f0_up_key,
        f0_method,
        file_index,
        # file_big_npy,
        index_rate,
        if_f0,
        filter_radius,
        tgt_sr,
        resample_sr,
        rms_mix_rate,
        version,
        f0_file=f0_file,
        protect=protect,
        ret_audio_opt=True,
    )
    return audio_opt


def get_vc(model_path):
    global n_spk, tgt_sr, net_g, vc, cpt, device, is_half, version
    # print("loading pth %s" % model_path)
    cpt = torch.load(model_path, map_location="cpu")
    tgt_sr = cpt["config"][-1]
    cpt["config"][-3] = cpt["weight"]["emb_g.weight"].shape[0]  # n_spk
    if_f0 = cpt.get("f0", 1)
    version = cpt.get("version", "v1")
    if version == "v1":
        if if_f0 == 1:
            net_g = SynthesizerTrnMs256NSFsid(*cpt["config"], is_half=is_half)
        else:
            net_g = SynthesizerTrnMs256NSFsid_nono(*cpt["config"])
    elif version == "v2":
        if if_f0 == 1:
            net_g = SynthesizerTrnMs768NSFsid(*cpt["config"], is_half=is_half)
        else:
            net_g = SynthesizerTrnMs768NSFsid_nono(*cpt["config"])
    del net_g.enc_q
    print(net_g.load_state_dict(cpt["weight"], strict=False))
    net_g.eval().to(device)
    if is_half:
        net_g = net_g.half()
    else:
        net_g = net_g.float()
    vc = VC(tgt_sr, config)
    n_spk = cpt["config"][-3]
    return model_path


# In[4]:


# 定义RVC一些参数
f0_up_key = 0
model_path = "weights/three_moon_e20_s10000.pth"
file_index = ''
f0_method = 'rmvpe'


# In[5]:


# 将音频信号转化回Mel频谱图
mel_spec = torchaudio.transforms.MelSpectrogram(
    sample_rate=24000,
    n_fft=1024,
    hop_length=256,
    n_mels=100,
    center=True,
    power=1,
)

def safe_log(x: torch.Tensor, clip_val: float = 1e-7) -> torch.Tensor:
    """
    Computes the element-wise logarithm of the input tensor with clipping to avoid near-zero values.

    Args:
        x (Tensor): Input tensor.
        clip_val (float, optional): Minimum value to clip the input tensor. Defaults to 1e-7.

    Returns:
        Tensor: Element-wise logarithm of the input tensor with clipping applied.
    """
    return torch.log(torch.clip(x, min=clip_val))


# In[6]:


import librosa
import torchaudio
import numpy as np
from scipy.io.wavfile import write
import matplotlib.pyplot as plt

def synthesize_and_process_audio(text):
    """
    整合函数，用于生成文本对应的语音，通过RVC换声，并将结果转换为梅尔频谱图。
    
    参数:
    - text: 输入文本字符串
    
    返回:
    - hidden: ChatTTS生成的隐藏特征
    - log_mel_spec: 经过RVC处理的音频的梅尔频谱图
    """
    # 使用ChatTTS生成语音
    torch.manual_seed(SEED)
    chat_result = chat.infer_debug(text=text, params_infer_code=params_infer_code)
    audio_numpy = chat_result['wav'][0]
    hidden = chat_result['hiddens'][0]
    
    # 重采样至16kHz
    resample_audio = librosa.resample(audio_numpy, orig_sr=24000, target_sr=16000)[0]
    
    # 通过RVC换声
    audio_opt = vc_single(
        sid=0,
        audio=resample_audio,
        f0_up_key=f0_up_key,
        f0_method=f0_method,
        file_index=file_index,
        filter_radius=3,
        resample_sr=24000,
        rms_mix_rate=1,
        model_path=model_path,
        protect=0.33,
    )
    
    # 将转换后的音频转换为梅尔频谱图
    log_mel_spec = safe_log(mel_spec(torch.from_numpy(audio_opt)))
    
    return hidden, log_mel_spec.numpy()


# In[7]:


import hashlib
import os
from tqdm import tqdm
from pathlib import Path

def compute_md5(text):
    """计算文本的MD5"""
    return hashlib.md5(text.encode('utf-8')).hexdigest()

def save_data(hidden, log_mel_spec, md5):
    """保存数据到指定目录"""
    save_folder = "train_rvc"
    os.makedirs(save_folder, exist_ok=True)
    filename = os.path.join(save_folder, f"{md5}.npz")
    if not os.path.exists(filename):
        np.savez(filename, hidden=hidden, log_mel_spec=log_mel_spec)

def process_texts_from_file(file_path):
    """从文件中读取文本，处理并保存数据"""
    with open(file_path, 'r', encoding='utf-8') as file:
        lines = file.readlines()
    
    for line in tqdm(lines, desc="Processing texts", unit="line"):
        line = line.strip()  # 移除末尾的空白字符
        
        # 跳过太小的句子
        if len(line) < 10:
            continue
        md5 = compute_md5(line)
        save_path = os.path.join("train_rvc", f"{md5}.npz")
        
        # 检查文件是否已存在，如果存在则跳过
        if os.path.exists(save_path):
            continue
        
        hidden, log_mel_spec = synthesize_and_process_audio(line)
        save_data(hidden, log_mel_spec, md5)


# In[8]:


# 假设你的文本文件路径是固定的或者作为参数传递
file_path = "StarRail_labs.txt"
process_texts_from_file(file_path)

