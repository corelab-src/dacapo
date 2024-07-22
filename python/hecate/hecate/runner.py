

import ctypes
import weakref
import re
import inspect
from subprocess import Popen
from collections.abc import Iterable
# import torch

import os
import time
import numpy as np
import numpy.ctypeslib as npcl
from pathlib import Path

import json


hecate_dir = Path(os.environ["HECATE"])
hecateBuild = hecate_dir / "build" 


if not hecateBuild.is_dir() : # We expect that this is library path 
    hecateBuild  = hecate_dir

libpath = hecateBuild / "lib"
lw = ctypes.CDLL(libpath / "libSEAL_HEVM.so")
# lw = ctypes.CDLL(libpath / "libHEAAN_HEVM.so")
# lw = ctypes.CDLL(libpath / "libTOY_HEVM.so")
os.environ['PATH'] = str(libpath) + os.pathsep + os.environ['PATH']


# Init VM functions
lw.initFullVM.argtypes = [ctypes.c_char_p, ctypes.c_bool]
lw.initFullVM.restype = ctypes.c_void_p 
lw.initClientVM.argtypes = [ctypes.c_char_p]
lw.initClientVM.restype = ctypes.c_void_p 
lw.initServerVM.argtypes = [ctypes.c_char_p]
lw.initServerVM.restype = ctypes.c_void_p 

# Init SEAL Contexts
lw.create_context.argtypes = [ctypes.c_char_p]
lw.load.argtypes = [ctypes.c_void_p, ctypes.c_char_p, ctypes.c_char_p]
lw.loadClient.argtypes = [ctypes.c_void_p,  ctypes.c_void_p]
lw.getArgLen.argtypes = [ctypes.c_void_p]
lw.getArgLen.restype = ctypes.c_int64
lw.getResLen.argtypes = [ctypes.c_void_p]
lw.getResLen.restype = ctypes.c_int64

# Encrypt/Decrypt Functions
lw.encrypt.argtypes = [ctypes.c_void_p, ctypes.c_int64, ctypes.POINTER(ctypes.c_double), ctypes.c_int]
lw.decrypt.argtypes = [ctypes.c_void_p, ctypes.c_int64, ctypes.POINTER(ctypes.c_double)]
lw.decrypt_result.argtypes = [ctypes.c_void_p, ctypes.c_int64, ctypes.POINTER(ctypes.c_double)]

# Helper Functions for ciphertext access
lw.getResIdx.argtypes = [ctypes.c_void_p, ctypes.c_int64]
lw.getResIdx.restype = ctypes.c_int64 
lw.getCtxt.argtypes = [ctypes.c_void_p, ctypes.c_int64]
lw.getCtxt.restype = ctypes.c_void_p 

# Runner Functions
lw.preprocess.argtypes = [ctypes.c_void_p]
lw.run.argtypes = [ctypes.c_void_p]

#Debug Function
lw.setDebug.argtypes = [ctypes.c_void_p, ctypes.c_bool]

#ToGPU Function
lw.setToGPU.argtypes = [ctypes.c_void_p, ctypes.c_bool]
lw.printMem.argtypes = [ctypes.c_void_p]

def reinit_lw():
    global lw
    if(run_library == "SEAL"):
        lw = ctypes.CDLL(libpath / "libSEAL_HEVM.so")
    elif(run_library == "HEAAN"):
        lw = ctypes.CDLL(libpath / "libHEAAN_HEVM.so")
    elif(run_library == "OPENFHE"):
        lw = ctypes.CDLL(libpath / "libOPENFHE_HEVM.so")
    elif(run_library == "TOY"):
        lw = ctypes.CDLL(libpath / "libTOY_HEVM.so")

    # Init VM functions
    lw.initFullVM.argtypes = [ctypes.c_char_p, ctypes.c_bool]
    lw.initFullVM.restype = ctypes.c_void_p 
    lw.initClientVM.argtypes = [ctypes.c_char_p]
    lw.initClientVM.restype = ctypes.c_void_p 
    lw.initServerVM.argtypes = [ctypes.c_char_p]
    lw.initServerVM.restype = ctypes.c_void_p 

    # Init SEAL Contexts
    lw.create_context.argtypes = [ctypes.c_char_p]
    lw.load.argtypes = [ctypes.c_void_p, ctypes.c_char_p, ctypes.c_char_p]
    lw.loadClient.argtypes = [ctypes.c_void_p,  ctypes.c_void_p]
    lw.getArgLen.argtypes = [ctypes.c_void_p]
    lw.getArgLen.restype = ctypes.c_int64
    lw.getResLen.argtypes = [ctypes.c_void_p]
    lw.getResLen.restype = ctypes.c_int64

    # Encrypt/Decrypt Functions
    lw.encrypt.argtypes = [ctypes.c_void_p, ctypes.c_int64, ctypes.POINTER(ctypes.c_double), ctypes.c_int]
    lw.decrypt.argtypes = [ctypes.c_void_p, ctypes.c_int64, ctypes.POINTER(ctypes.c_double)]
    lw.decrypt_result.argtypes = [ctypes.c_void_p, ctypes.c_int64, ctypes.POINTER(ctypes.c_double)]

    # Helper Functions for ciphertext access
    lw.getResIdx.argtypes = [ctypes.c_void_p, ctypes.c_int64]
    lw.getResIdx.restype = ctypes.c_int64 
    lw.getCtxt.argtypes = [ctypes.c_void_p, ctypes.c_int64]
    lw.getCtxt.restype = ctypes.c_void_p 

    # Runner Functions
    lw.preprocess.argtypes = [ctypes.c_void_p]
    lw.run.argtypes = [ctypes.c_void_p]

    #Debug Function
    lw.setDebug.argtypes = [ctypes.c_void_p, ctypes.c_bool]



run_library="TOY"
run_hardware="GPU"
def setLibnHW (argv=None):
    global run_library
    global run_hardware
    LibnHW_mapping = {
            # "HEAAN" : ["GPU", "CPU"],
            # "SEAL" : ["CPU"],
            # "TOY" : ["CPU", "GPU"],
            }
    HWnLib_mapping = {}
    for support_libs, support_HWs in LibnHW_mapping.items():
        for HW in support_HWs:
            if HW in HWnLib_mapping:
                HWnLib_mapping[HW].append(support_libs)
            else:
                HWnLib_mapping[HW] = [support_libs]
    # TODO : How about using getopt
    if len(argv) >= 4:
        if(argv[3].upper() in LibnHW_mapping.keys()):
            run_library = argv[3].upper()
            if len(argv) == 5:
                if(argv[4].upper() in LibnHW_mapping[run_library]):
                    run_hardware = argv[4].upper()
                else:
                    print("Not supported",argv[4].upper())
                    print("Suppoerted hardware :", LibnHW_mapping[run_library])
                    exit()
            else:
                run_hardware = LibnHW_mapping[run_library][0]
        elif(argv[3].upper() in HWnLib_mapping.keys()):
            run_hardware = argv[3].upper()
            if len(argv) == 5:
                if(argv[4].upper() in HWnLib_mapping[run_hardware]):
                    run_library = argv[4].upper()
                else:
                    print("Not supported",argv[4].upper())
                    print("Suppoerted library :", HWnLib_mapping[run_hardware])
                    exit()        
            else:
                run_library = HWnLib_mapping[run_hardware][0]
        else:
            print("Not supported",argv[3])
            print("Require Bootstrapping-supported Library")
            print("Supported library :",LibnHW_mapping.keys())
            print("Supported hardware :",HWnLib_mapping.keys())
            exit()        
    # else:
        #        # For default
#        run_library = list(LibnHW_mapping.keys())[0]
#        run_hardware = LibnHW_mapping[run_library][0]


class HEVM : 
    def __init__ (self, path = str((Path.home() / ".hevm" / "heaan").absolute()) , option= "full") :
        global run_library
        global run_hardware
        reinit_lw()

        self.option = option
        if(run_library == "SEAL"):
            if(path == str((Path.home() / ".hevm" / "heaan").absolute())):
                # If path is default
                path = str((Path.home() / ".hevm" / "seal").absolute())
        if not Path(path).is_dir() : 
            if(run_library == "SEAL"):
                print ("Press Any key to generate SEAL files (or just kill with ctrl+c)")
            elif(run_library == "HEAAN"):
                print ("Press Any key to generate HEaaN files (or just kill with ctrl+c)")
            input()
            Path(path).mkdir(parents=True)
            lw.create_context(path.encode('utf-8'))

        if option == "full" :
            if(run_hardware == "GPU"):
                self.vm = lw.initFullVM(path.encode('utf-8'), True)
            elif(run_hardware == "CPU"):
                self.vm = lw.initFullVM(path.encode('utf-8'), False)
        elif option == "client" :
            self.vm = lw.initClientVM(path.encode('utf-8'))
        elif  option == "server" :
            self.vm = lw.initServerVM(path.encode('utf-8'))

    # def load (self, func,   preprocess=True, const_path =str( (Path(func_dir) / "_hecate_{func}.cst").absoluate() ), hevm_path = str(Path(func_dir) / "_hecate_{func}.hevm"), func_dir = str(Path.cwd()), ) :
    def load (self, const_path, hevm_path, preprocess=True) :
        if not Path(const_path).is_file() :
            raise Exception(f"No file exists in const_path {const_path}")
        if not Path(hevm_path).is_file() :
            raise Exception(f"No file exists in hevm_path {hevm_path}")
        if self.option == "full" or self.option == "server" :
            lw.load(self.vm, const_path.encode('utf-8'), hevm_path.encode('utf-8'))
        elif self.option ==  "client" :
            lw.loadClient (self.vm, const_path.encode('utf-8'))
        if (preprocess) :
            lw.preprocess (self.vm)
        else :
            raise Exception("Not implemented in SEAL_HEVM")

        self.arglen = lw.getArgLen(self.vm)
        self.reslen = lw.getResLen(self.vm)
        self.hevm_path = hevm_path

    def run (self) : 
        lw.run(self.vm)
        lw.printMem(self.vm)

    def setInput(self, i, data) :
        if not isinstance(data, np.ndarray) :
            data = np.array(data, dtype=np.float64)
        carr = data.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
        lw.encrypt(self.vm, i, carr, len(data))

    def setDebug (self, enable) : 
        lw.setDebug(self.vm, enable)

    def setToGPU (self, ongpu) :
        lw.setToGPU(self.vm, ongpu)

    def getOutput (self) : 
        if(run_library == "HEAAN"):
            result = np.zeros( (self.reslen, 1 << 16), dtype=np.float64)
            data = np.zeros(  1 << 16, dtype=np.float64)
            # result = np.zeros( (self.reslen, 1 << 14), dtype=np.float64)
            # data = np.zeros(  1 << 14, dtype=np.float64)
        elif(run_library == "SEAL"):
            result = np.zeros( (self.reslen, 1 << 14), dtype=np.float64)
            data = np.zeros(  1 << 14, dtype=np.float64)
        for i in range(self.reslen) :
            # carr = npcl.as_ctypes(data) 
            carr =  data.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
            lw.decrypt_result(self.vm, i, carr)
            # result[i] = npcl.as_array(carr, shape= 1<<14)
            result[i] = data
        return result
    
    def printer(self, latency, rms, mem_usage = 0.0) :
        import re
        bench = re.search(r"optimized/(.*)/(.*)\.(.*)\._", self.hevm_path)
        print("======================================")
        print("---------------Option-----------------")
        print("compiler:", bench.group(1))
        print("benchname:", bench.group(2))
        print("waterline:", bench.group(3))
        print("library:", run_library)
        print("device:", run_hardware)
        print("---------------Result-----------------")
        print("latency:", latency)
        print("rms:", rms)
        # print("memory_usage:", mem_usage)
        print("======================================")
        print()


