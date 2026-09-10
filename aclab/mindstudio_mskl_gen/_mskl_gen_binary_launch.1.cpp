
#include <cstdlib>
#include <climits>
#include <iostream>
#include <vector>
#include <unordered_map>
#include <fstream>
#include <getopt.h>
#include <Python.h>
#include <dlfcn.h>
#include <sys/stat.h>
#include <sys/wait.h>
#include "acl/acl.h"

using namespace std;

constexpr int rtDevBinaryMagicElf = 0x43554245U;
constexpr int rtDevBinaryMagicElfAivec = 0x41415246U;
constexpr int rtDevBinaryMagicElfAicube = 0x41494343U;
#define CHECK_RT_RESULT(result) if (!(result)) {return false;}
#define LOG(__level, __msg, ...) printf(__level __msg "\n", ##__VA_ARGS__)
#define LOGI(__msg, ...) LOG("[INFO ] ", __msg, ##__VA_ARGS__)
#define LOGW(__msg, ...) LOG("[WARN ] ", __msg, ##__VA_ARGS__)
#define LOGE(__msg, ...) LOG("[ERROR] ", __msg, ##__VA_ARGS__)

typedef uint32_t rtError_t;
typedef void *rtStream_t;

typedef struct tagRtDevBinary {
    uint32_t magic;
    uint32_t version;
    const void *data;
    uint64_t length;
} rtDevBinary_t;

typedef struct tagRtSmData {
    uint64_t L2_mirror_addr;          // preload or swap source addr
    uint32_t L2_data_section_size;    // every data size
    uint8_t L2_preload;               // 1 - preload from mirrorAddr, 0 - no preload
    uint8_t modified;                 // 1 - data will be modified by kernel, 0 - no modified
    uint8_t priority;                 // data priority
    int8_t prev_L2_page_offset_base;  // remap source section offset
    uint8_t L2_page_offset_base;      // remap destination section offset
    uint8_t L2_load_to_ddr;           // 1 - need load out, 0 - no need
    uint8_t reserved[2];              // reserved
} rtSmData_t;

typedef struct rtHostInputInfo {
    uint32_t addrOffset;
    uint32_t dataOffset;
} rtHostInputInfo_t;

typedef struct tagRtArgsEx {
    void *args;                     // args host mem addr
    rtHostInputInfo_t *hostInputInfoPtr;     // nullptr means no host mem input
    uint32_t argsSize;              // input + output + tiling addr size + tiling data size + host mem
    uint32_t tilingAddrOffset;      // tiling addr offset
    uint32_t tilingDataOffset;      // tiling data offset
    uint16_t hostInputInfoNum;      // hostInputInfo num
    uint8_t hasTiling;              // if has tiling: 0 means no tiling
    uint8_t isNoNeedH2DCopy;        // is no need host to device copy: 0 means need H2D copy,
    // others means doesn't need H2D copy.
    uint8_t reserved[4];
} rtArgsEx_t;

typedef struct tagRtTaskCfgInfo {
    uint8_t qos;
    uint8_t partId;
    uint8_t schemMode; // rtschemModeType_t 0:normal;1:batch;2:sync
    uint8_t res[1]; // res
} rtTaskCfgInfo_t;

typedef struct tagRtSmCtrl {
    rtSmData_t data[8];  // data description
    uint64_t size;       // max page Num
    uint8_t remap[64];   /* just using for static remap mode, default:0xFF
                            array index: virtual l2 page id, array value: physic l2 page id */
    uint8_t l2_in_main;  // 0-DDR, 1-L2, default:0xFF
    uint8_t reserved[3];
} rtSmDesc_t;

struct OpgenKernelConfig {
    uint64_t tilingKey {0};
    int blockDim {0};
    void *stream{nullptr};
    string kernelBinaryPath; // .o路径
    vector<void *> kernelArgs;
};

extern "C" {
rtError_t rtDevBinaryUnRegister(void *hdl);
rtError_t rtRegisterAllKernel(const rtDevBinary_t *bin, void **hdl);
rtError_t rtKernelLaunchWithHandleV2(void *hdl, const uint64_t tilingKey, uint32_t blockDim,
                                     rtArgsEx_t *argsInfo, rtSmDesc_t *smDesc, rtStream_t stm,
                                     const rtTaskCfgInfo_t *cfgInfo);
rtError_t rtGetC2cCtrlAddr(uint64_t *addr, uint32_t *fftsLen);
rtError_t rtGetSocVersion(char *version, const uint32_t maxLen);
}

namespace Adx {
void AdumpPrintWorkSpace(const void *workSpaceAddr, const size_t dumpWorkSpaceSize,
                         rtStream_t stream, const char *opType);
}

size_t GetFileSize(const string &filePath)
{
    struct stat fileStat;
    if (stat(filePath.c_str(), &fileStat) != 0 || !S_ISREG(fileStat.st_mode)) {
        return 0;
    }
    return static_cast<size_t>(fileStat.st_size);
}

size_t ReadBinary(string const &filename, vector<char> &data)
{
    ifstream ifs(filename, ios::binary);
    if (!ifs.is_open()) {
        return 0;
    }
    ifs.seekg(0, ifstream::end);
    int64_t length = ifs.tellg();
    if (length < 0) {
        return 0;
    }
    ifs.seekg(0, ifstream::beg);
    data.resize(length);
    ifs.read(data.data(), length);
    return length;
}

bool PipeCall(vector<string> const &cmd, string &output)
{
    int pipeStdout[2];
    if (pipe(pipeStdout) != 0) {
        return false;
    }
    pid_t pid = fork();
    if (pid < 0) {
        return false;
    } else if (pid == 0) {
        unsetenv("LD_PRELOAD");
        dup2(pipeStdout[1], STDOUT_FILENO);
        dup2(STDOUT_FILENO, STDERR_FILENO);
        close(pipeStdout[0]);
        close(pipeStdout[1]);

        std::vector<char *> rawArgv;
        for (auto const &arg: cmd) {
            rawArgv.emplace_back(const_cast<char *>(arg.data()));
        }
        rawArgv.emplace_back(nullptr);
        execvp(cmd[0].c_str(), rawArgv.data());
        _exit(EXIT_FAILURE);
    } else {
        close(pipeStdout[1]);

        constexpr std::size_t bufLen = 256UL;
        char buf[bufLen] = {'\0'};
        ssize_t nBytes = 0L;
        for (; (nBytes = read(pipeStdout[0], buf, bufLen)) > 0L;) {
            output.append(buf, static_cast<std::size_t>(nBytes));
        }
        close(pipeStdout[0]);

        int status;
        waitpid(pid, &status, 0);
        return WIFEXITED(status) && WEXITSTATUS(status) == 0;
    }
}

class KernelRunner {
public:
    KernelRunner()
    {
        constexpr uint64_t socVersionBufLen = 64UL;
        char socVersion[socVersionBufLen] = "";
        if (CheckRtResult(rtGetSocVersion(socVersion, sizeof(socVersion)), "rtGetSocVersion")) {
            soc = socVersion;
            if (soc.empty()) {
                LOGE("rtGetSocVersion failed, soc-version is empty");
            }
        }
        if (magic == 0) {
            string fixStr = "input [kernel_type] in [mskl.get_kernel_from_binary] manually.";
            if (!soc.empty() && soc.find("Ascend310P") != string::npos) {
                LOGI("Set kernel_type as mix, you can change this value by %s", fixStr.c_str());
                magic = rtDevBinaryMagicElf;
            } else {
                string mixTag = "_mix_";
                vector<string> cmd = {"llvm-objdump", "-t", "/workspace/kda/kda_bt16/aclab/build/k2dev_aiv_device_dir/device_aiv.o"};
                string output;
                if (!PipeCall(cmd, output)) {
                    LOGE("Get magic from [kernel_binary_file] failed, please %s", fixStr.c_str());
                } else if (output.find(mixTag, output.find("SYMBOL TABLE:")) != string::npos) { // 避免文件名的影响
                    LOGI("Set kernel_type as mix, you can change this value by %s", fixStr.c_str());
                    magic = rtDevBinaryMagicElf;
                } else {
                    LOGI("Set kernel_type as vec, you can change this value by %s", fixStr.c_str());
                    magic = rtDevBinaryMagicElfAivec;
                }
            }
        }
    }

    bool PyRun(const OpgenKernelConfig& config)
    {
        if (!needUnRegisterDevBinary_) {
            // register kernel
            size_t fileSize = GetFileSize(config.kernelBinaryPath);
            vector<char> bin;
            if (ReadBinary(config.kernelBinaryPath, bin) == 0) {
                return false;
            }
            if (!RegisterKernel(config, bin, fileSize)) {
                return false;
            }
        }
        if (!LaunchKernel(config)) {
            return false;
        }
        return true;
    }

    ~KernelRunner()
    {
        if (needUnRegisterDevBinary_) {
            CheckRtResult(rtDevBinaryUnRegister(binHandle_), "rtDevBinaryUnRegister");
        }
    }

private:
    bool RegisterKernel(const OpgenKernelConfig &kernelConfig, const vector<char> &data, uint64_t fileSize)
    {
        rtDevBinary_t deviceBinary {};
        deviceBinary.version = 0;
        deviceBinary.data = data.data();
        deviceBinary.magic = magic;
        deviceBinary.length = fileSize;
        CHECK_RT_RESULT(CheckRtResult(rtRegisterAllKernel(&deviceBinary, &binHandle_), "rtRegisterAllKernel"));
        needUnRegisterDevBinary_ = true;
        return true;
    }

    bool LaunchKernel(const OpgenKernelConfig &kernelConfig)
    {
        kernelArgs_.clear();
        if (soc.find("Ascend910B") != string::npos && (magic != rtDevBinaryMagicElfAivec)) {
            // 910B非vec算子，需要在第一个入参传入ffts地址
            InitFftsAddr();
        }


        if (soc.find("Ascend310P") != string::npos) {
            // 310P关闭overflow
            int err = aclrtSetStreamOverflowSwitch(kernelConfig.stream, 0);
            if (err != 0) {
                LOGE("Call aclrtSetStreamOverflowSwitch failed, error code: %d", err);
                return false;
            }
        }

        kernelArgs_.insert(kernelArgs_.end(), kernelConfig.kernelArgs.begin(), kernelConfig.kernelArgs.end());
        rtArgsEx_t argsEx {};
        argsEx.args = kernelArgs_.data();
        argsEx.hostInputInfoPtr = nullptr;
        argsEx.argsSize = kernelArgs_.size() * sizeof(void*);
        argsEx.hasTiling = 0;
        argsEx.isNoNeedH2DCopy = 0; // args指针本身指向host，依然需要h2dcopy
        CHECK_RT_RESULT(CheckRtResult(
            rtKernelLaunchWithHandleV2(binHandle_, kernelConfig.tilingKey,
                                       kernelConfig.blockDim, &argsEx, nullptr,
                                       kernelConfig.stream, nullptr),
            "rtKernelLaunchWithHandleV2"));

        if (0) {
            if (kernelArgs_.size() < 2) {
                LOGW("kernel args smaller than 2, disable ASCENDC::printf ability.");
                return true;
            }
            void *workspace = kernelArgs_[kernelArgs_.size() - 2U];
            uint32_t debugBufferSize = 75 * 1024 * 1024;
            Adx::AdumpPrintWorkSpace(workspace, debugBufferSize, kernelConfig.stream, "manual");
        }
        return true;
    }

    bool InitFftsAddr()
    {
        uint64_t addr;
        uint32_t addrLen;
        CHECK_RT_RESULT(CheckRtResult(rtGetC2cCtrlAddr(&addr, &addrLen), "rtGetC2cCtrlAddr"))
        kernelArgs_.emplace_back(reinterpret_cast<void *>(addr));
        return true;
    }

    bool CheckRtResult(rtError_t result, const string &apiName)
    {
        if (result == 0) {
            return true;
        }
        LOGE("Runtime API call %s() failed. error code: %d", apiName.c_str(), result);
        return false;
    }

    void *binHandle_ = nullptr;
    bool needUnRegisterDevBinary_ = false;
    vector<void *> kernelArgs_;
    string soc;
    uint32_t magic = 1094799942;
};

static PyObject* _launch_kernel(PyObject* self, PyObject* args)
{
    Py_ssize_t size = PyTuple_Size(args);
    if (size < 3) {
        // 至少传入blockdim, l2ctrl, stream 3个入参
        std::string errorStr = "size of args is " + std::to_string(size) + ". it must be not less than 3";
        PyErr_SetString(PyExc_ValueError, errorStr.c_str());
        Py_RETURN_NONE;
    }
    std::vector<void *> pyArgs;
    pyArgs.reserve(size);

    for (Py_ssize_t i = 0; i < size; i++) {
        PyObject* item = PyTuple_GetItem(args, i);
        if (!PyLong_Check(item)) {
            PyErr_SetString(PyExc_TypeError, "All arguments must be integers");
            Py_RETURN_NONE;
        }
        void *temp = PyLong_AsVoidPtr(item);
        pyArgs.push_back(temp);
    }

    OpgenKernelConfig kernelConfig;
    kernelConfig.tilingKey = 0;
    kernelConfig.blockDim = (uint64_t)pyArgs[0];
    kernelConfig.stream = pyArgs[2];

    kernelConfig.kernelBinaryPath = "/workspace/kda/kda_bt16/aclab/build/k2dev_aiv_device_dir/device_aiv.o";
    std::vector<void *> kernelArgs(pyArgs.begin() + 3, pyArgs.end());
    kernelConfig.kernelArgs = kernelArgs;
    static KernelRunner kernelRunner;

    kernelRunner.PyRun(kernelConfig);
    Py_RETURN_NONE;
}

static PyMethodDef ModuleMethods[] = {
    {"kernel_binary", _launch_kernel, METH_VARARGS, "Entry point for kernel kernel_binary"},
    {NULL, NULL, 0, NULL}
};

static struct PyModuleDef ModuleDef = {
    PyModuleDef_HEAD_INIT,
    "_mskl_launcher",
    NULL,
    -1,
    ModuleMethods
};

#if PY_VERSION_HEX < 0x03090000
extern "C" __attribute__((visibility("default"))) PyObject* PyInit__mskl_launcher(void)
#else
PyMODINIT_FUNC PyInit__mskl_launcher(void)
#endif
{
    PyObject *m = PyModule_Create(&ModuleDef);
    if(m == NULL) {
        return NULL;
    }
    PyModule_AddFunctions(m, ModuleMethods);
    return m;
}
