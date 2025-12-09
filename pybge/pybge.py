import ctypes
import os
from typing import List, Tuple
import numpy as np
import platform
from pyaxdev import _lib, AxDeviceType, AxDevices, check_error


class BgeInit(ctypes.Structure):
    _fields_ = [
        ('dev_type', AxDeviceType),
        ('devid', ctypes.c_int),
        ('filename_axmodel', ctypes.c_char * 1024),
        ('tokenizer_model', ctypes.c_char * 1024),
    ]

class BgeEmbeding(ctypes.Structure):
    _fields_ = [
        ('len_of_tokens', ctypes.c_int),
        ('embeding', ctypes.c_float * 384)
    ]


_lib.ax_embeding_init.argtypes = [ctypes.POINTER(BgeInit), ctypes.POINTER(ctypes.c_void_p)]
_lib.ax_embeding_init.restype = ctypes.c_int

_lib.ax_embeding_deinit.argtypes = [ctypes.c_void_p]
_lib.ax_embeding_deinit.restype = ctypes.c_int

_lib.ax_embeding.argtypes = [ctypes.c_void_p, ctypes.c_char_p, ctypes.POINTER(BgeEmbeding)]
_lib.ax_embeding.restype = ctypes.c_int

_lib.ax_similarity.argtypes = [ctypes.POINTER(BgeEmbeding), ctypes.POINTER(BgeEmbeding)]
_lib.ax_similarity.restype = ctypes.c_float


class Bge:
    def __init__(self, init_info: dict):
        self.handle = None
        self.init_info = BgeInit()
        
        # 设置初始化参数
        self.init_info.dev_type = init_info.get('dev_type', AxDeviceType.axcl_device)
        self.init_info.devid = init_info.get('devid', 0)
        
        # 设置路径
        for path_name in ['filename_axmodel', 'tokenizer_model']:
            if path_name in init_info:
                setattr(self.init_info, path_name, init_info[path_name].encode('utf-8'))
        
        # 创建CLIP实例
        handle = ctypes.c_void_p()
        check_error(_lib.ax_embeding_init(ctypes.byref(self.init_info), ctypes.byref(handle)))
        self.handle = handle

    def __del__(self):
        if self.handle:
            _lib.ax_embeding_deinit(self.handle)

    def embed(self, text: str) -> None:
        embeding = BgeEmbeding()
        check_error(_lib.ax_embeding(self.handle, text.encode('utf-8'), ctypes.byref(embeding)))
        return np.array(embeding.embeding[:384], dtype=np.float32)
        
        

    def similarity(self, embeding1: np.ndarray, embeding2: np.ndarray) -> float:
        embeding1_ = BgeEmbeding()
        for i in range(384):
            embeding1_.embeding[i] = embeding1[i]
        embeding2_ = BgeEmbeding()
        for i in range(384):
            embeding2_.embeding[i] = embeding2[i]
        return _lib.ax_similarity(ctypes.byref(embeding1_), ctypes.byref(embeding2_))
