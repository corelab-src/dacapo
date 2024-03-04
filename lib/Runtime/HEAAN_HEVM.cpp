#include <HEaaN/Context.hpp>
#include <HEaaN/Message.hpp>
#include <HEaaN/Plaintext.hpp>
#include <HEaaN/device/CudaTools.hpp>
#include <HEaaN/device/Device.hpp>
#include <any>
#include <cassert>
#include <chrono>
#include <fstream>
#include <iostream>

#include <HEaaN/HEaaN.hpp>
#include <HEaaN/ParameterPreset.hpp>
#include <cmath>
#include <map>

#include <type_traits>
#include <vector>

#include "hecate/Support/HEVMHeader.h"

struct HEAAN_HEVM {
  std::vector<std::vector<double>> buffer;
  HEVMHeader header;
  ConfigBody config;
  /* std::vector<uint64_t> config_dats; */
  std::vector<HEVMOperation> ops;
  std::vector<uint64_t> arg_scale;
  std::vector<uint64_t> arg_level;
  std::vector<uint64_t> res_scale;
  std::vector<uint64_t> res_level;
  std::vector<uint64_t> res_dst;

  std::vector<HEaaN::Ciphertext> ciphers;
  std::vector<double> scalec;
  std::vector<HEaaN::Plaintext> plains;
  std::vector<double> scalep;
  std::vector<uint64_t> levelp;
  std::vector<HEaaN::Message> msgs;
  std::map<double *, int> msgMap;
  std::map<uint64_t, HEaaN::Plaintext> upscale_const;

  HEaaN::Context context;
  std::unique_ptr<HEaaN::KeyPack> keypack;
  std::unique_ptr<HEaaN::SecretKey> seckey;
  std::unique_ptr<HEaaN::Encryptor> encryptor;
  std::unique_ptr<HEaaN::HomEvaluator> evaluator;
  std::unique_ptr<HEaaN::Bootstrapper> bootstrapper;
  std::unique_ptr<HEaaN::Decryptor> decryptor;
  std::unique_ptr<HEaaN::EnDecoder> endecoder;
  /* std::chrono::microseconds boot_time; */
  uint64_t boot_time = 0;
  uint64_t boot_cnt = 0;

  static const int N = 17;
  static const int L = 16;

  std::vector<int64_t> rotKeyOffset = {
      1,     2,     3,     4,     5,     6,     7,     8,     16,    24,
      32,    64,    96,    128,   160,   192,   224,   256,   512,   768,
      1024,  2048,  3072,  4096,  5120,  6144,  7168,  8192,  16384, 24576,
      32768, 40960, 49152, 57344, 61440, 63488, 64512, 64768, 65024, 65280,
      65408, 65472, 65504, 65512, 65520, 65528, 65532, 65534, 65535,
  };
  /* std::vector<int64_t> rotKeyOffset = { */
  /*     1,     2,     3,     4,     5,     6,     7,     8,     16, */
  /*     24,    32,    64,    96,    128,   160,   192,   224,   256, */
  /*     512,   768,   1024,  2048,  3072,  4096,  5120,  6144,  7168, */
  /*     8192,  9216,  10240, 11264, 12288, 13312, 14336, 15360, 15616, */
  /*     15872, 16192, 16224, 16256, 16288, 16320, 16352, 16360, 16368, */
  /*     16376, 16377, 16378, 16379, 16380, 16381, 16382, 16383}; */

  bool debug = false;
  bool togpu = true;
  bool preencode = false;

  static void create_context(char *dir) {

    auto strdir = std::string(dir);

    /* HEaaN::ParameterPreset preset = HEaaN::ParameterPreset::ST19; */
    HEaaN::ParameterPreset preset = HEaaN::ParameterPreset::FVa;
    auto context = HEaaN::makeContext(preset, {0});
    auto num_full_slot = getLogFullSlots(context);
    HEaaN::SecretKey sk(context);
    {
      std::ofstream f(strdir + "/sec.heaan", std::ios::out | std::ios::binary);
      sk.save(f);
      f.close();
    }
    HEaaN::KeyPack kp(context);
    HEaaN::KeyGenerator keygen(context, sk, kp);
    keygen.genCommonKeys();
    keygen.genRotKeysForBootstrap(num_full_slot);
    {
      keygen.save(strdir);
      HEaaN::saveContextToFile(context, strdir + "/context.heaan");
    }
  }
  void printCudaMemInfo() {
    auto MemUse = HEaaN::CudaTools::getCudaMemoryInfo().second -
                  HEaaN::CudaTools::getCudaMemoryInfo().first;
    auto TotalMemCapacity = HEaaN::CudaTools::getCudaMemoryInfo().second;
    std::cout << "MemUsage: " << MemUse / std::pow(10, 9) << "GB";
    std::cout << " (" << double(MemUse * 100) / double(TotalMemCapacity) << "%)"
              << "\n";
  }
  void printInfo() {
    int encodeOnline = 2;
    /* std::cout << "polyDegree: " << N << '\n'; */
    /* std::cout << "encodeOnline: " << encodeOnline << "\n\n"; */
  }

  void loadHEAAN(char *dir) {
    HEaaN::setUVM(HEaaN::getCurrentCudaDevice(), false);
    auto strdir = std::string(dir);
    {
      context = HEaaN::makeContextFromFile(strdir + "/context.heaan", {0});
      seckey =
          std::make_unique<HEaaN::SecretKey>(context, strdir + "/sec.heaan");
      keypack = std::make_unique<HEaaN::KeyPack>(context, strdir);
      keypack->loadEncKey();
      keypack->loadMultKey();
      for (auto offset : rotKeyOffset) {
        keypack->loadLeftRotKey(offset);
      }
    }
    encryptor = std::make_unique<HEaaN::Encryptor>(context);
    decryptor = std::make_unique<HEaaN::Decryptor>(context);
    endecoder = std::make_unique<HEaaN::EnDecoder>(context);
    evaluator = std::make_unique<HEaaN::HomEvaluator>(context, *keypack);
    bootstrapper = std::make_unique<HEaaN::Bootstrapper>(*evaluator);
    if (togpu) {
      /* printCudaMemInfo(); */
      seckey->to(HEaaN::getCurrentCudaDevice());
      keypack->to(HEaaN::getCurrentCudaDevice());
      bootstrapper->makeBootConstants(HEaaN::getLogFullSlots(context));
      bootstrapper->loadBootConstants(HEaaN::getLogFullSlots(context),
                                      HEaaN::getCurrentCudaDevice());
      /* printCudaMemInfo(); */
    }
  }

  void loadClient(char *dir) {
    auto strdir = std::string(dir);
    context = HEaaN::makeContextFromFile(strdir + "/context.heaan", {0});

    encryptor = std::make_unique<HEaaN::Encryptor>(context);
    decryptor = std::make_unique<HEaaN::Decryptor>(context);
    endecoder = std::make_unique<HEaaN::EnDecoder>(context);
  }

  void loadServer(char *dir) {
    auto strdir = std::string(dir);
    context = HEaaN::makeContextFromFile(strdir + "/context.heaan", {0});
    keypack = std::make_unique<HEaaN::KeyPack>(context, strdir);

    endecoder = std::make_unique<HEaaN::EnDecoder>(context);
    evaluator = std::make_unique<HEaaN::HomEvaluator>(context, *keypack);
    bootstrapper = std::make_unique<HEaaN::Bootstrapper>(*evaluator);
  }

  void loadConstants(char *name) {
    std::string sname(name);

    std::ifstream iff(sname, std::ios::binary);
    int64_t len;
    iff.read((char *)&len, sizeof(int64_t));
    buffer.resize(len);

    for (int64_t i = 0; i < len; i++) {
      int64_t veclen;
      iff.read((char *)&veclen, sizeof(int64_t));
      std::vector<double> tmp;
      tmp.resize(veclen);
      iff.read((char *)tmp.data(), veclen * sizeof(double));
      buffer[i] = tmp;
    }
    iff.close();
    /* std::cerr << "Constant Loaded From" << sname << std::endl; */
  }

  void loadHEVM(char *name) {
    std::string sname(name);

    std::ifstream iff(sname, std::ios::binary);

    loadHeader(iff);

    ops.resize(config.num_operations);
    iff.read((char *)ops.data(), ops.size() * sizeof(HEVMOperation));

    ciphers.resize(config.num_ctxt_buffer, HEaaN::Ciphertext(context));
    if (togpu) {
      for (auto &&cipher : ciphers)
        cipher.to(HEaaN::getCurrentCudaDevice());
    }

    HEaaN::u64 log_slot = N - 1;
    HEaaN::Message datas(log_slot, 0.0);
    msgs.resize(config.num_ptxt_buffer, datas);
    if (preencode) {
      plains.resize(config.num_ptxt_buffer, HEaaN::Plaintext(context));
      if (togpu) {
        for (auto &&plain : plains)
          plain.to(HEaaN::getCurrentCudaDevice());
      }
    } else {
      plains.resize(1, HEaaN::Plaintext(context));
      if (togpu) {
        plains[0].to(HEaaN::getCurrentCudaDevice());
      }
    }
    scalec.resize(config.num_ctxt_buffer);
    scalep.resize(config.num_ptxt_buffer);
    levelp.resize(config.num_ptxt_buffer);
  }

  void loadHeader(std::istream &iff) {

    iff.read((char *)&header, sizeof(HEVMHeader));
    iff.read((char *)&config, sizeof(ConfigBody));

    arg_scale.resize(header.config_header.arg_length);
    arg_level.resize(header.config_header.arg_length);
    res_scale.resize(header.config_header.res_length);
    res_level.resize(header.config_header.res_length);
    res_dst.resize(header.config_header.res_length);
    iff.read((char *)arg_scale.data(), arg_scale.size() * sizeof(uint64_t));
    iff.read((char *)arg_level.data(), arg_level.size() * sizeof(uint64_t));
    iff.read((char *)res_scale.data(), res_scale.size() * sizeof(uint64_t));
    iff.read((char *)res_level.data(), res_level.size() * sizeof(uint64_t));
    iff.read((char *)res_dst.data(), res_dst.size() * sizeof(uint64_t));

    ciphers.resize(header.config_header.arg_length +
                       header.config_header.res_length,
                   HEaaN::Ciphertext(context));
    scalec.resize(config.num_ctxt_buffer);
  }

  void resetResDst() {
    for (size_t i = 0; i < header.config_header.res_length; i++) {
      res_dst[i] = i + header.config_header.arg_length;
    }
  }

  void preprocess() {
    std::vector<double> identity(1LL << (N - 1), 1.0);
    for (HEVMOperation &op : ops) {
      if (op.opcode == 0) {
        if (preencode) {
          encode_internal(plains[op.dst],
                          op.lhs == ((unsigned short)-1) ? identity
                                                         : buffer[op.lhs],
                          op.rhs >> 10, op.rhs & 0x3FF);
        } else {
          to_msg(op.dst,
                 op.lhs == ((unsigned short)-1) ? identity : buffer[op.lhs]);
        }
        levelp[op.dst] = op.rhs >> 10;
        scalep[op.dst] = op.rhs & 0x3FF;
      }
    }
  }

  void to_msg(int16_t dst, std::vector<double> src) {
    auto &msg = msgs[dst];
    for (size_t i = 0; i < msg.getSize(); i++) {
      msg[i].real(src[i % src.size()]);
      msg[i].imag(0);
    }
    if (togpu)
      msg.to(HEaaN::getCurrentCudaDevice());
    return;
  }
  void encode_online(int16_t dst) {
    /* if (togpu) */
    /*   msgs[dst].to(HEaaN::getCurrentCudaDevice()); */
    plains[0] =
        endecoder->encode(msgs[dst], levelp[dst], std::pow(2.0, scalep[dst]));
  }

  void encode_internal(HEaaN::Plaintext &dst, std::vector<double> src,
                       int8_t level, uint64_t scale) {
    HEaaN::u64 log_slot = N - 1;
    HEaaN::Message datas(log_slot, 0.0);

    for (size_t i = 0; i < datas.getSize(); i++) {
      datas[i].real(src[i % src.size()]);
      datas[i].imag(0);
    }
    if (togpu) {
      datas.to(HEaaN::getCurrentCudaDevice());
    }
    dst = endecoder->encode(datas, level, std::pow(2.0, scale));
    return;
  }

  void encode(int16_t dst, int16_t src, int8_t level, int8_t scale) { return; }
  void rotate(int16_t dst, int16_t src, int16_t offset) {
    if (debug)
      std::cout << scalec[src] << std::endl;
    evaluator->leftRotate(ciphers[src], offset, ciphers[dst]);
    scalec[dst] = scalec[src];
  }
  void negate(int16_t dst, int16_t src) {
    if (debug) {
      std::cout << scalec[src] << std::endl;
    }
    evaluator->negate(ciphers[src], ciphers[dst]);
    scalec[dst] = scalec[src];
  }
  void rescale(int16_t dst, int16_t src) {
    if (debug)
      std::cout << scalec[src] << std::endl;
    ciphers[dst] = ciphers[src];
    scalec[dst] =
        scalec[src] - std::round(ciphers[src].getCurrentScaleFactor());
    ciphers[dst].setRescaleCounter(1);
    evaluator->rescale(ciphers[dst]);
  }
  void modswitch(int16_t dst, int16_t src, int16_t downFactor) {
    if (debug) {
      std::cout << scalec[src] << " " << downFactor << std::endl;
      std::cout << "before level : " << ciphers[src].getLevel() << std::endl;
    }
    if (downFactor > 0) {
      scalec[dst] =
          scalec[src] - std::round(ciphers[src].getCurrentScaleFactor());
      evaluator->levelDownOne(ciphers[src], ciphers[dst]);
      scalec[dst] += std::round(ciphers[dst].getCurrentScaleFactor());
    }
    for (int i = 1; i < downFactor; i++) {
      scalec[dst] =
          scalec[dst] - std::round(ciphers[dst].getCurrentScaleFactor());
      evaluator->levelDownOne(ciphers[dst], ciphers[dst]);
      scalec[dst] += std::round(ciphers[dst].getCurrentScaleFactor());
    }
    if (debug) {
      std::cout << scalec[dst] << " " << downFactor << std::endl;
      std::cout << "after level : " << ciphers[dst].getLevel() << std::endl;
    }
  }
  void upscale(int16_t dst, int16_t src, int16_t upFactor) {
    assert(0 && "This VM does not support native upscale op");
  }
  void addcc(int16_t dst, int16_t lhs, int16_t rhs) {
    if (debug)
      std::cout << scalec[lhs] << " " << scalec[rhs] << std::endl;
    scalec[dst] = scalec[lhs];
    evaluator->add(ciphers[lhs], ciphers[rhs], ciphers[dst]);
  }
  void addcp(int16_t dst, int16_t lhs, int16_t rhs) {
    if (debug)
      std::cout << scalec[lhs] << " " << scalep[rhs] << std::endl;
    scalec[dst] = scalec[lhs];
    if (preencode) {
      evaluator->add(ciphers[lhs], plains[rhs], ciphers[dst]);
    } else {
      encode_online(rhs);
      evaluator->add(ciphers[lhs], plains[0], ciphers[dst]);
    }
  }
  void mulcc(int16_t dst, int16_t lhs, int16_t rhs) {
    if (debug)
      std::cout << scalec[lhs] << " " << scalec[rhs] << std::endl;
    evaluator->multWithoutRescale(ciphers[lhs], ciphers[rhs], ciphers[dst]);
    ciphers[dst].setRescaleCounter(0);
    scalec[dst] = scalec[lhs] + scalec[rhs];
  }
  void mulcp(int16_t dst, int16_t lhs, int16_t rhs) {
    if (debug) {
      std::cout << scalec[lhs] << " " << scalep[rhs] << std::endl;
      std::cout << "cipher level : " << ciphers[lhs].getLevel() << '\n';
      std::cout << "plain level : " << levelp[rhs] << '\n';
    }
    if (preencode) {
      evaluator->multWithoutRescale(ciphers[lhs], plains[rhs], ciphers[dst]);
    } else {
      encode_online(rhs);
      evaluator->multWithoutRescale(ciphers[lhs], plains[0], ciphers[dst]);
    }
    ciphers[dst].setRescaleCounter(0);
    scalec[dst] = scalec[lhs] + scalep[rhs];
  }
  void bootstrap(int16_t dst, int64_t src, uint64_t targetLevel) {
    if (debug) {
      std::cout << scalec[src] << " " << ciphers[src].getLevel() << std::endl;
    }
    auto time_start = std::chrono::high_resolution_clock::now();
    bootstrapper->bootstrap(ciphers[src], ciphers[dst], targetLevel, false);
    HEaaN::CudaTools::cudaDeviceSynchronize();
    auto time_end = std::chrono::high_resolution_clock::now();
    auto time_diff = std::chrono::duration_cast<std::chrono::microseconds>(
        time_end - time_start);
    boot_time += time_diff.count();
    boot_cnt++;
    scalec[dst] = ciphers[dst].getCurrentScaleFactor();
  }

  void run() {
    int i = (header.hevm_header_size + config.config_body_length) / 8;
    int j = 0;
    /* HEaaN::CudaTools::cudaDeviceSynchronize(); */
    for (HEVMOperation &op : ops) {
      if (debug) {
        std::cout << std::endl;
        std::cout << std::oct << i++ << " " << std::dec << j++ << std::endl;
        std::cout << "opcode [" << op.opcode << "], dst [" << op.dst
                  << "], lhs [" << op.lhs << "], rhs [" << op.rhs << "]"
                  << std::endl;
      }
      switch (op.opcode) {
      case 0: { // Encode
        encode(op.dst, op.lhs, op.rhs >> 10, op.rhs & 0x3FF);
        break;
      }
      case 1: { // RotateC
        rotate(op.dst, op.lhs, op.rhs);
        break;
      }
      case 2: { // NegateC
        negate(op.dst, op.lhs);
        break;
      }
      case 3: { // RescaleC
        rescale(op.dst, op.lhs);
        break;
      }
      case 4: { // ModswtichC
        modswitch(op.dst, op.lhs, op.rhs);
        break;
      }
      case 5: { // UpscaleC
        upscale(op.dst, op.lhs, op.rhs);
        break;
      }
      case 6: { // AddCC
        addcc(op.dst, op.lhs, op.rhs);
        break;
      }
      case 7: { // AddCP
        addcp(op.dst, op.lhs, op.rhs);
        break;
      }
      case 8: { // MulCC
        mulcc(op.dst, op.lhs, op.rhs);
        break;
      }
      case 9: { // MulCP
        mulcp(op.dst, op.lhs, op.rhs);
        break;
      }
      case 10: { // Bootstrap
        HEaaN::CudaTools::cudaDeviceSynchronize();
        bootstrap(op.dst, op.lhs, op.rhs);
        break;
      }
      default: {
        break;
      }
      }
    }
    /* std::cout << "boot_time : " << boot_time << '\n'; */
    /* std::cout << "boot_cnt : " << boot_cnt << '\n'; */
  }
};

extern "C" {
void *initFullVM(char *dir, bool device = false) {
  auto vm = new HEAAN_HEVM();
  vm->togpu = device;
  vm->loadHEAAN(dir);
  return (void *)vm;
}
void *initClientVM(char *dir) {
  auto vm = new HEAAN_HEVM();
  vm->loadClient(dir);
  return (void *)vm;
}
void *initServerVM(char *dir) {
  auto vm = new HEAAN_HEVM();
  vm->loadServer(dir);
  return (void *)vm;
}

void create_context(char *dir) { HEAAN_HEVM::create_context(dir); }

// Loader for server
void load(void *vm, char *constant, char *vmfile) {
  auto hevm = static_cast<HEAAN_HEVM *>(vm);
  hevm->loadConstants(constant);
  hevm->loadHEVM(vmfile);
}

// Loader for client
void loadClient(void *vm, void *is) {
  auto hevm = static_cast<HEAAN_HEVM *>(vm);
  std::istream &iss = *static_cast<std::istream *>(is);
  hevm->loadHeader(iss);
  hevm->resetResDst();
}

// encryption and decryption uses internal buffer id
void encrypt(void *vm, int64_t i, double *dat, int len) {
  auto hevm = static_cast<HEAAN_HEVM *>(vm);
  HEaaN::Plaintext ptxt(hevm->context);
  std::vector<double> dats(dat, dat + len);
  hevm->encode_internal(ptxt, dats, hevm->arg_level[i], hevm->arg_scale[i]);
  hevm->encryptor->encrypt(ptxt, *hevm->seckey, hevm->ciphers[i]);
  if (hevm->togpu) {
    hevm->ciphers[i].to(HEaaN::getCurrentCudaDevice());
  }
  hevm->scalec[i] = hevm->arg_scale[i];
}
void decrypt(void *vm, int64_t i, double *dat) {
  auto hevm = static_cast<HEAAN_HEVM *>(vm);
  HEaaN::Plaintext ptxt(hevm->context);
  hevm->decryptor->decrypt(hevm->ciphers[i], *hevm->seckey, ptxt);
  HEaaN::Message msg =
      hevm->endecoder->decode(ptxt, std::pow(2.0, std::round(hevm->scalec[i])));
  if (hevm->togpu)
    msg.to(HEaaN::getDefaultDevice());
  for (int i = 0; i < (1LL << (HEAAN_HEVM::N - 1)); i++) {
    /* for (size_t j = 0; j < msg.getSize(); j++) */
    dat[i] = msg[i].real();
  }
}
// simple wrapper to elide getResIdx call
// use res_idx for i
void decrypt_result(void *vm, int64_t i, double *dat) {
  auto hevm = static_cast<HEAAN_HEVM *>(vm);
  decrypt(vm, hevm->res_dst[i], dat);
}

// We need this for communication code to access the proper buffer id
int64_t getResIdx(void *vm, int64_t i) {
  auto hevm = static_cast<HEAAN_HEVM *>(vm);
  return hevm->res_dst[i];
}

// use this to implement communication
void *getCtxt(void *vm, int64_t id) {
  auto hevm = static_cast<HEAAN_HEVM *>(vm);
  return &(hevm->ciphers[id]);
}

void preprocess(void *vm) {
  auto hevm = static_cast<HEAAN_HEVM *>(vm);
  hevm->preprocess();
}
void run(void *vm) {
  auto hevm = static_cast<HEAAN_HEVM *>(vm);
  hevm->run();
}
int64_t getArgLen(void *vm) {
  auto hevm = static_cast<HEAAN_HEVM *>(vm);
  return hevm->header.config_header.arg_length;
}
int64_t getResLen(void *vm) {
  auto hevm = static_cast<HEAAN_HEVM *>(vm);
  return hevm->header.config_header.res_length;
}
void setDebug(void *vm, bool enable) {
  auto hevm = static_cast<HEAAN_HEVM *>(vm);
  hevm->debug = enable;
}
void setToGPU(void *vm, bool ongpu) {
  auto hevm = static_cast<HEAAN_HEVM *>(vm);
  hevm->togpu = ongpu;
}
void printMem(void *vm) {
  auto hevm = static_cast<HEAAN_HEVM *>(vm);
  hevm->printCudaMemInfo();
  hevm->printInfo();
}
};
