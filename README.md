# hecate-compiler
Hecate (Homomorphic Encryption Compiler for Approximate TEnsor computation) is an optimizing compiler for the CKKS FHE scheme, built by Compiler Optimization Research Laboratory (Corelab) @ Yonsei University. 
Hecate is built on the top of Multi-Level Intermediate Representation (MLIR) compiler framework. 
We aim to support privacy-preserving machine learning and deep learning applications. 


  * [Installation](#installation)
    + [Requirements](#requirements)
    + [Install MLIR](#install-mlir)
    + [Install SEAL](#install-seal)
    + [Build Hecate](#build-hecate)
    + [Configure Hecate](#configure-hecate)
    + [Install Hecate Python Binding](#install-hecate-python-binding)
  * [Tutorial](#tutorial)
    + [Trace the example python file to Encrypted ARiTHmetic IR](#trace-the-example-python-file-to-encrypted-arithmetic-ir)
    + [Compile the traced Earth Hecate IR](#compile-the-traced-earth-ir)
    + [Compile the traced Earth Hecate IR and Check the Optimized Code](#compile-the-traced-earth-ir-and-check-the-optimized-code)
    <!-- + [Test the optimized code](#test-the-optimized-code) -->
    <!-- + [One-liner for compilation and testing](#one-liner-for-compilation-and-testing) -->
  * [Papers](#papers)
  * [Citations](#citations)

## Installation 

### Requirements 
```
Ninja   
git  
cmake >= 3.22.1  
python >= 3.10  
clang,clang++ >= 14.0.0  
```

### Install MLIR 
```bash
git clone https://github.com/llvm/llvm-project.git
cd llvm-project
git checkout llvmorg-18.1.2
cmake -GNinja -Bbuild \
  -DCMAKE_C_COMPILER=clang \
  -DCMAKE_CXX_COMPILER=clang++ -DCMAKE_BUILD_TYPE=Release \
  -DLLVM_ENABLE_PROJECTS=mlir -DLLVM_INSTALL_UTILS=ON \
  -DLLVM_TARGETS_TO_BUILD=host \
  llvm
cmake --build build
sudo cmake --install build
cd .. 
```
#### Optional : Install Directory  to maintain multiple versions or a debug build 
```bash
git clone https://github.com/llvm/llvm-project.git
cd llvm-project
git checkout llvmorg-18.0.2
cmake -GNinja -Bbuild \ 
  -DCMAKE_C_COMPILER=clang \
  -DCMAKE_CXX_COMPILER=clang++ -DCMAKE_BUILD_TYPE=Release\
  -DLLVM_ENABLE_PROJECTS=mlir -DLLVM_INSTALL_UTILS=ON \
  -DLLVM_TARGETS_TO_BUILD=host -DCMAKE_INSTALL_PREFIX=<MLIR_INSTALL>\
  llvm
cmake --build build
sudo cmake --install build
cd .. 
```

### Install SEAL 
```bash
git clone https://github.com/microsoft/SEAL.git
cd SEAL
git checkout 4.0.0
cmake -S . -B build
cmake --build build
sudo cmake --install build
cd .. 
```
#### Optional : Install Directory  to maintain multiple versions or a debug build
```bash
git clone https://github.com/microsoft/SEAL.git
cd SEAL
git checkout 4.0.0
cmake -S . -B build -DCMAKE_INSTALL_PREFIX=<SEAL_INSTALL>
cmake --build build
sudo cmake --install build
cd .. 
```
### Build Hecate 
```bash
git clone <this-repository>
cd <this-repository>
cmake -S . -B build 
cmake --build build 
```
#### Optional : Install Directory  to maintain multiple versions or a debug build
```bash
git clone <this-repository>
cd <this-repository>
cmake -S . -B build -DMLIR_ROOT=<MLIR_INSTALL> -DSEAL_ROOT=<SEAL_INSTALL>
cmake --build build 
```
### Configure Hecate 
```bash
python3 -m venv .venv
source .venv/bin/activate
source config.sh 
```

### Install Hecate Python Binding 
```bash
pip install -r requirements.txt
./install.sh
```

## Tutorial 

### Trace the example python file to Encrypted ARiTHmetic IR 

```bash
hc-trace <example-name>
```
e.g., 
```bash
hc-trace ResNet
```

### Compile the traced Earth IR 

```bash
hopts <pars|dacapo> <waterline:integer> <example-name>
```
e.g., 
```bash
hopts dacapo 40 ResNet
```
This command will print like this:
```
Estimated Latency: 13.348852 (sec)
Number of Bootstrapping: 19
```

### Compile the traced Earth IR and Check the optimized code 
```bash
hbt <pars|dacapo> <waterline:integer> <example-name>
```
e.g., 
```bash
hbt dacapo 40 ResNet
```
You can see the optimized code in "$hecate-compiler/examples/optimized/dacapo/ResNet.40.earth.mlir"\

If you see an error message like "error: 'earth.bootstrap' op failed to infer returned types",\
just wait as it is in the normal compilation process.

## Papers 
**DaCapo: Automatic Bootstrapping Management for Efficient Fully Homomorphic Encryption**\
Seonyoung Cheon, Yongwoo Lee, Ju Min Lee, Dongkwan Kim, Sunchul Jung, Taekyung Kim, Dongyoon Lee, and Hanjun Kim  
*33rd USENIX Security Symposium (USENIX Security)*, August 2024. 
[[Prepublication](https://www.usenix.org/system/files/sec24summer-prepub-336-cheon.pdf)]

**ELASM: Error-Latency-Aware Scale Management for Fully Homomorphic Encryption** [[abstract](https://www.usenix.org/conference/usenixsecurity23/presentation/lee-yongwoo)]   
Yongwoo Lee, Seonyoung Cheon, Dongkwan Kim, Dongyoon Lee, and Hanjun Kim  
*32nd USENIX Security Symposium (USENIX Security)*, August 2023. 
[[Prepublication](https://www.usenix.org/system/files/sec23fall-prepub-147-lee-yongwoo.pdf)]

**HECATE: Performance-Aware Scale Optimization for Homomorphic Encryption Compiler**\[[IEEE Xplore](http://doi.org/10.1109/CGO53902.2022.9741265)]   
Yongwoo Lee, Seonyeong Heo, Seonyoung Cheon, Shinnung Jeong, Changsu Kim, Eunkyung Kim, Dongyoon Lee, and Hanjun Kim  
*Proceedings of the 2022 International Symposium on Code Generation and Optimization (CGO)*, April 2022. 
[[Prepublication](http://corelab.or.kr/Pubs/cgo22_hecate.pdf)]

## Citations 
```bibtex
@INPROCEEDINGS{lee:hecate:cgo,
  author={Lee, Yongwoo and Heo, Seonyeong and Cheon, Seonyoung and Jeong, Shinnung and Kim, Changsu and Kim, Eunkyung and Lee, Dongyoon and Kim, Hanjun},
  booktitle={2022 IEEE/ACM International Symposium on Code Generation and Optimization (CGO)}, 
  title={HECATE: Performance-Aware Scale Optimization for Homomorphic Encryption Compiler}, 
  year={2022},
  volume={},
  number={},
  pages={193-204},
  doi={10.1109/CGO53902.2022.9741265}}
```
```bibtex
@INPROCEEDINGS{lee:elasm:sec,
  title={{ELASM}: Error-Latency-Aware Scale Management for Fully Homomorphic Encryption},
  author={Lee, Yongwoo and Cheon, Seonyoung and Kim, Dongkwan and Lee, Dongyoon and Kim, Hanjun},
  booktitle={{32nd} USENIX Security Symposium (USENIX Security 23)},
 year={2023},
 address = {Anaheim, CA},
 publisher = {USENIX Association},
 month = aug
}
```
```bibtex
@INPROCEEDINGS{cheon:dacapo:sec,
  title={{DaCapo}: Automatic Bootstrapping Management for Efficient Fully Homomorphic Encryption},
  author={Cheon, Seonyoung and Lee, Yongwoo and Kim, Dongkwan and Lee, Ju Min and Jung, Sunchul and Kim, Taekyung and Lee, Dongyoon and Kim, Hanjun},
  booktitle={{33rd} USENIX Security Symposium (USENIX Security 24)},
 year={2024},
 address = {Philadelphia, CA},
 publisher = {USENIX Association},
 month = aug
}
```

