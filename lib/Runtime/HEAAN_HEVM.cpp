#include <HEaaN/Context.hpp>
#include <HEaaN/device/Device.hpp>
#include <cassert>
#include <fstream>
#include <iostream>

#include <HEaaN/HEaaN.hpp>
#include <HEaaN/ParameterPreset.hpp>
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
  std::map<uint64_t, HEaaN::Plaintext> upscale_const;

  HEaaN::Context context;
  std::unique_ptr<HEaaN::KeyPack> keypack;
  std::unique_ptr<HEaaN::SecretKey> seckey;
  std::unique_ptr<HEaaN::Encryptor> encryptor;
  std::unique_ptr<HEaaN::HomEvaluator> evaluator;
  std::unique_ptr<HEaaN::Bootstrapper> bootstrapper;
  std::unique_ptr<HEaaN::Decryptor> decryptor;
  std::unique_ptr<HEaaN::EnDecoder> endecoder;

  static const int N = 17;
  static const int L = 16;

  bool debug = false;
  bool togpu = false;

  static void create_context(char *dir) {

    auto strdir = std::string(dir);

    HEaaN::ParameterPreset preset = HEaaN::ParameterPreset::FVa;
    auto context = HEaaN::makeContext(preset, {0});
    HEaaN::SecretKey sk(context);
    {
      std::ofstream f(strdir + "/sec.heaan", std::ios::out | std::ios::binary);
      sk.save(f);
      f.close();
    }
    HEaaN::KeyPack kp(context);
    HEaaN::KeyGenerator keygen(context, sk, kp);
    keygen.genCommonKeys();
    {
      keygen.save(strdir);
      HEaaN::saveContextToFile(context, strdir + "/context.heaan");
    }
  }

  void loadHEAAN(char *dir) {
    auto strdir = std::string(dir);
    {
      context = HEaaN::makeContextFromFile(strdir + "/context.heaan", {0});
      seckey =
          std::make_unique<HEaaN::SecretKey>(context, strdir + "/sec.heaan");
      keypack = std::make_unique<HEaaN::KeyPack>(context, strdir);
    }
    encryptor = std::make_unique<HEaaN::Encryptor>(context);
    decryptor = std::make_unique<HEaaN::Decryptor>(context);
    endecoder = std::make_unique<HEaaN::EnDecoder>(context);
    evaluator = std::make_unique<HEaaN::HomEvaluator>(context, *keypack);
    bootstrapper = std::make_unique<HEaaN::Bootstrapper>(*evaluator);
    if (togpu) {
      seckey->to(HEaaN::getCurrentCudaDevice());
      keypack->to(HEaaN::getCurrentCudaDevice());
      bootstrapper->loadBootConstants(HEaaN::getLogFullSlots(context),
                                      HEaaN::getCurrentCudaDevice());
      /* HEaaN::setUVM(HEaaN::getCurrentCudaDevice(), true); */
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
    plains.resize(config.num_ptxt_buffer, HEaaN::Plaintext(context));

    scalec.resize(config.num_ctxt_buffer);
    scalep.resize(config.num_ptxt_buffer);
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
        encode_internal(plains[op.dst],
                        op.lhs == ((unsigned short)-1) ? identity
                                                       : buffer[op.lhs],
                        op.rhs >> 8, op.rhs & 0xFF);
        scalep[op.dst] = op.rhs & 0xFF;
      }
    }
  }

  void encode_internal(HEaaN::Plaintext &dst, std::vector<double> src,
                       int8_t level, int8_t scale) {
    HEaaN::u64 log_slot = N - 1;
    HEaaN::Message datas(log_slot, 0.0);

    for (size_t i = 0; i < datas.getSize(); i++) {
      datas[i].real(src[i % src.size()]);
      datas[i].imag(0);
    }
    if (togpu)
      datas.to(HEaaN::getCurrentCudaDevice());
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
    scalec[dst] = scalec[src] - ciphers[src].getCurrentScaleFactor();
    ciphers[dst].setRescaleCounter(1);
    evaluator->rescale(ciphers[dst]);
  }
  void modswitch(int16_t dst, int16_t src, int16_t downFactor) {
    if (debug)
      std::cout << scalec[src] << std::endl;
    if (downFactor > 0) {
      scalec[dst] = scalec[src] - ciphers[src].getCurrentScaleFactor();
      evaluator->levelDownOne(ciphers[src], ciphers[dst]);
      scalec[dst] += ciphers[dst].getCurrentScaleFactor();
    }
    for (int i = 1; i < downFactor; i++) {
      scalec[dst] = scalec[dst] - ciphers[dst].getCurrentScaleFactor();
      evaluator->levelDownOne(ciphers[dst], ciphers[dst]);
      scalec[dst] += ciphers[dst].getCurrentScaleFactor();
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
    evaluator->add(ciphers[lhs], plains[rhs], ciphers[dst]);
  }
  void mulcc(int16_t dst, int16_t lhs, int16_t rhs) {
    if (debug)
      std::cout << scalec[lhs] << " " << scalec[rhs] << std::endl;
    evaluator->multWithoutRescale(ciphers[lhs], ciphers[rhs], ciphers[dst]);
    ciphers[dst].setRescaleCounter(0);
    scalec[dst] = scalec[lhs] + scalec[rhs];
  }
  void mulcp(int16_t dst, int16_t lhs, int16_t rhs) {
    if (debug)
      std::cout << scalec[lhs] << " " << scalep[rhs] << std::endl;
    evaluator->multWithoutRescale(ciphers[lhs], plains[rhs], ciphers[dst]);
    ciphers[dst].setRescaleCounter(0);
    scalec[dst] = scalec[lhs] + scalep[rhs];
  }
  void bootstrap(int16_t dst, int64_t src, int8_t targetLevel) {
    if (debug)
      std::cout << scalec[src] << std::endl;
    bootstrapper->bootstrap(ciphers[src], ciphers[dst], targetLevel, false);
    scalec[dst] = ciphers[dst].getCurrentScaleFactor();
  }

  void run() {
    int i = (header.hevm_header_size + config.config_body_length) / 8;
    int j = 0;
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
        encode(op.dst, op.lhs, op.rhs >> 8, op.rhs & 0xFF);
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
        bootstrap(op.dst, op.lhs, op.rhs);
        break;
      }
      default: {
        /* assert(0 && "Invalid opcode"); */
        break;
      }
      }
    }
  }
};

extern "C" {
void *initFullVM(char *dir) {
  auto vm = new HEAAN_HEVM();
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
  hevm->scalec[i] = hevm->arg_scale[i];
}
void decrypt(void *vm, int64_t i, double *dat) {
  auto hevm = static_cast<HEAAN_HEVM *>(vm);
  HEaaN::Plaintext ptxt(hevm->context);
  hevm->decryptor->decrypt(hevm->ciphers[i], *hevm->seckey, ptxt);
  HEaaN::Message msg =
      hevm->endecoder->decode(ptxt, std::pow(2.0, hevm->scalec[i]));
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
};
