# bge-small-en-v1.5

## 模型
[AXERA-TECH/bge-small-en-v1.5](https://huggingface.co/AXERA-TECH/bge-small-en-v1.5)

## 编译
### x86构建

```bash
git clone --recursive https://github.com/AXERA-TECH/bge.axera.git
cd bge.axera
sudo apt install build-essential 
./build.sh
```

### AArch64构建

#### 交叉编译aarch64

```bash
git clone --recursive https://github.com/AXERA-TECH/bge.axera.git
cd bge.axera
./build_aarch64.sh
```

#### 在目标板上原生构建

```bash
git clone --recursive https://github.com/AXERA-TECH/bge.axera.git
cd bge.axera
sudo apt install build-essential
./build.sh
```
---


## 运行
```shell
./test_model -m bge-small-en-v1.5_u16_npu3.axmodel
open libax_sys.so failed
open libax_engine.so failed
[I][                             run][  31]: AXCLWorker start with devid 0
filename_axmodel: bge-small-en-v1.5_u16_npu3.axmodel

input size: 1
    name: input_ids [unknown] [unknown] 
        1 x 512  2048


output size: 1
    name: embeddings 
        1 x 512 x 384  786432

tokenizer_type = 2
similarity between [I really love math] and [I pretty like mathematics] is 0.883220
similarity between [I really love math] and [same as me] is 0.557530
similarity between [so do I] and [I pretty like mathematics] is 0.643069
similarity between [so do I] and [same as me] is 0.735331
[I][                             run][  81]: AXCLWorker exit with devid 0
```

## api
```shell
$ ./main_api -m bge-small-en-v1.5_u16_npu3.axmodel --port 28080
open libax_sys.so failed
open libax_engine.so failed
[I][                             run][  31]: AXCLWorker start with devid 0
filename_axmodel: bge-small-en-v1.5_u16_npu3.axmodel

input size: 1
    name: input_ids [unknown] [unknown] 
        1 x 512  2048


output size: 1
    name: embeddings 
        1 x 512 x 384  786432

tokenizer_type = 2
listen http://0.0.0.0:28080
for example: 
    curl -X POST http://0.0.0.0:28080/embedding -H "Content-Type: application/json" -d '{"sentence":"I really love math"}'
press Ctrl+C to stop
```
```shell
$ curl -X POST http://0.0.0.0:28080/embedding -H "Content-Type: application/json" -d '{"sentence":"I really love math"}'

{"embedding":[-0.020170118659734726,-0.0010176385985687375,0.0056....
```


## 社区
QQ 群: 139953715