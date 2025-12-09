from pybge import Bge
from pyaxdev import enum_devices, sys_init, sys_deinit, AxDeviceType
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='../build/bge-small-en-v1.5_u16_npu3.axmodel')
    parser.add_argument('--tokenizer', type=str, default='../tests/tokenizer/bge_tokenizer.txt')
    args = parser.parse_args()

    # 枚举设备
    devices_info = enum_devices()
    dev_type = AxDeviceType.axcl_device
    devid = 0
    print("可用设备:", devices_info)
    if devices_info['host']['available']:
        print("host device available")
        sys_init(AxDeviceType.host_device, -1)
        dev_type = AxDeviceType.host_device
        devid = -1
    elif devices_info['devices']['count'] > 0:
        print("axcl device available, use device-0")
        sys_init(AxDeviceType.axcl_device, 0)
        dev_type = AxDeviceType.axcl_device
        devid = 0
    else:
        raise Exception("No available device")

    try:
        bge = Bge({
            'filename_axmodel': args.model,
            'tokenizer_model': args.tokenizer,
            'dev_type': dev_type,
            'devid': devid,
        })
        
        sentences_1 = "I really love math"
        sentences_2 = "I pretty like mathematics"
        
        embeddings_1 = bge.embed(sentences_1)
        embeddings_2 = bge.embed(sentences_2)
        print(embeddings_1.shape)
        print(embeddings_2.shape)
        
        print(bge.similarity(embeddings_1, embeddings_2))

        del bge
    finally:
        # 反初始化系统
        if devices_info['host']['available']:
            sys_deinit(AxDeviceType.host_device, -1)
        elif devices_info['devices']['count'] > 0:
            sys_deinit(AxDeviceType.axcl_device, devid)
