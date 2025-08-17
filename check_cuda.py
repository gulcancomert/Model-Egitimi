import torch

print("CUDA kullanılabilir mi?:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("CUDA cihaz adı:", torch.cuda.get_device_name(0))
    print("Toplam GPU sayısı:", torch.cuda.device_count())
