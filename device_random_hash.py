import torch
import cpuinfo

def hash_tensor(tensor):
    return hash(tuple(tensor.reshape(-1).tolist()))

processor_info = cpuinfo.get_cpu_info()
processor_name = processor_info.get('brand_raw')

# 定义设备列表
devices = [('cpu', processor_name)]
if torch.cuda.is_available():
    device_name = torch.cuda.get_device_name('cuda:0')
    devices.append(('cuda:0', device_name))
if torch.backends.mps.is_available():
    devices.append(('mps', processor_name))

# 在每个设备上生成随机数并计算哈希值
for device, device_name in devices:

    # 设置随机数种子
    torch.manual_seed(1234)

    # 生成随机数
    random_numbers = torch.rand(768, device=device)

    # 计算哈希值
    hash_value = hash_tensor(random_numbers)

    print(f'{device_name} 设备生成的随机数Hash是 {hash_value}')
