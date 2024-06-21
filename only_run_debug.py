import matplotlib.pyplot as plt

#模型下载
from modelscope import snapshot_download
model_dir = snapshot_download('mirror013/ChatTTS')

# 加载模型
import ChatTTS
chat = ChatTTS.Chat()
chat.load_models(
    source="local",
    local_path=model_dir,
    device='cpu',
    compile=False,
)

# # 推理并返回梅尔声谱图
# mel_spec = chat.infer(
#     text='你好，我是Chat T T S。',
#     use_decoder=True,
#     return_mel_spec=True,
# )

# # 绘制梅尔声谱图
# plt.figure(figsize=(10, 4))
# plt.imshow(mel_spec[0].detach().numpy(), aspect='auto', origin='lower')
# plt.title('Log-Mel Spectrogram')
# plt.xlabel('Time')
# plt.ylabel('Mel Frequency')
# plt.colorbar(format='%+2.0f dB')
# plt.show()


# 推理并且返回中间结果
infer_middle_result = chat.infer(
    text='你好，我是ChatTTS。', 
    skip_refine_text=True, 
    refine_text_only=True, 
    return_infer_token=True
)
print(infer_middle_result)