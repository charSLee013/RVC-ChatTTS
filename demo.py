import os
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
from IPython.display import Audio

torch._dynamo.config.cache_size_limit = 64
torch._dynamo.config.suppress_errors = True
torch.set_float32_matmul_precision('high')
logging.basicConfig(level=logging.DEBUG)

SEED = 1397

#模型下载
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

torch.manual_seed(SEED) # 音色种子
# load from local file if exists
if os.path.exists('spk_emb.npy'):
    spk_emb = torch.load('spk_emb.npy',map_location='cpu')
else:
    spk_emb = chat.sample_random_speaker()
params_infer_code = {
    'spk_emb':spk_emb,
    # "spk_emb":torch.randn(768),
    'temperature': 0.1,
    'top_P': 0.9,
    'top_K': 50,
}

# params_refine_text = {}
params_refine_text = {'prompt': '[oral_0][laugh_0][break_0]'}


texts = ["接下来,杨叔，借我看一下现场地图。他肯定穿过了前面的那扇门，不可能在这么小的地方晃悠了两小时。",
        #  "拿好留影机，我要出题了。",
        #  "嗯，复习一下构图要领，准备好再来吧。",
        #  "我这里没什么需要帮忙的，去看看其他人吧。",
        #  "你就安心吧。我的眼光能有错？",
        #  "克洛琳德说得没错啊，会长大人真是可靠！",
        #  "说起来，我以前也没想到克洛琳德小姐私下里讲话这么轻松呢！",
        #  "哈哈哈，跟你们聊天真开心，来翘英庄真是正确的选择。",
        ]

# 对文本进行预处理
new_texts = []


def filter_punctuation(text):
    allowed_punctuations = {".", ",", "!", "?", "，", "。", "！", "？"," "}
    new_text = ""
    for char in text:
        if char.isalnum() or char in allowed_punctuations:
            new_text += char
    return new_text

# 使用新函数替换原有的预处理步骤
for t in texts:
    filter_text = filter_punctuation(t)
    # 调用模型显示实际生存的文本
    filter_text:str = chat.infer(
        text=filter_text, skip_refine_text=False, refine_text_only=True,
        params_refine_text=params_refine_text,
        params_infer_code=params_infer_code,
        do_text_normalization=False)[0]
    # filter_text = filter_text.replace(" ","")
    logging.info(f"输入文本: {t}\n预处理后的文本: {filter_text}")
    new_texts.append(filter_text)

torch.manual_seed(SEED) # 推理种子
all_wavs = chat.infer([filter_punctuation(texts[0])], use_decoder=True,
                params_infer_code=params_infer_code,
                skip_refine_text=True,
                params_refine_text=params_refine_text,
                do_text_normalization=False)

# 确保所有数组的维度都是 (1, N)，然后进行合并
combined_wavs = np.concatenate(all_wavs, axis=1)

# audio_file = "./output.wav"
# # 将音频数据缩放到[-1, 1]范围内，这是wav文件的标准范围
# audio_data = combined_wavs / np.max(np.abs(combined_wavs))

# # 将浮点数转为16位整数，这是wav文件常用格式
# audio_data_int16 = (audio_data * 32767).astype(np.int16)

# # 保存到本地，注意采样率为24000
# with wave.open(audio_file, 'wb') as wf:
#     wf.setnchannels(1)
#     wf.setsampwidth(2)
#     wf.setframerate(24000)
#     wf.writeframes(audio_data_int16)

# # Save the generated audio 
torchaudio.save("output.wav", torch.from_numpy(combined_wavs), 24000)
