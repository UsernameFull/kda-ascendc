// Probe-only launcher shim: makes Ascend C RTC kernels visible to msopprof.
//
// msopprof's injection library (tools/msopprof/lib64/libmsopprof_injection.so)
// interposes the aclrtLaunchKernel* family and rtKernelLaunch, but *not*
// aclrtLaunchKernelWithArgsArray - which is the call the production launcher
// (aclab/launcher/launcher.cpp) uses for every kernel.  Profiling the
// production path therefore shows nothing: msopprof never sees the launch.
//
// This shim compiles the same kernel source through the same aclrtc options
// and launches it through aclrtLaunchKernel (which is interposed), with the
// kernel arguments packed into one host buffer - the form that call takes.
// It is deliberately a separate module so no production file changes and the
// production .so's build stays untouched.
//
//   python:  shim.rtc_handle(defines + source, name) -> handle
//            shim.launch(handle, blocks, packed_blob, stream)
#include <acl/acl.h>
#include <acl/acl_rt.h>
#include <acl/acl_rt_compile.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <cstdint>
#include <cstdlib>
#include <string>
#include <vector>

namespace py = pybind11;

namespace {

void Check(aclError err, const std::string &what) {
    if (err != ACL_SUCCESS) {
        throw std::runtime_error(what + " failed, aclError=" + std::to_string(err));
    }
}

void EnsureRt() {
    aclrtContext cur = nullptr;
    aclError e = aclrtGetCurrentContext(&cur);
    if (e != ACL_SUCCESS || cur == nullptr) {
        Check(aclInit(nullptr), "aclInit");
        Check(aclrtSetDevice(0), "aclrtSetDevice");
        aclrtContext ctx = nullptr;
        Check(aclrtCreateContext(&ctx, 0), "aclrtCreateContext");
        Check(aclrtSetCurrentContext(ctx), "aclrtSetCurrentContext");
    }
}

// Same option list as aclab/launcher/launcher.cpp so a shim-compiled kernel is
// the same device binary the production path would produce.
std::vector<std::string> CompileOptions() {
    const char *cannHomeEnv = std::getenv("ASCEND_HOME_PATH");
    const std::string cannHome =
        (cannHomeEnv != nullptr && cannHomeEnv[0] != '\0')
            ? cannHomeEnv
            : "/usr/local/Ascend/ascend-toolkit/latest";
    const std::string ascBase = cannHome +
#if defined(__aarch64__)
        "/aarch64-linux";
#else
        "/x86_64-linux";
#endif
    return {
        "--npu-soc=Ascend910B3", "-O3", "-std=c++17",
        "-I" + ascBase + "/tikcpp/tikcfw",
        "-I" + ascBase + "/tikcpp/tikcfw/interface",
        "-I" + ascBase + "/tikcpp/tikcfw/impl",
        "-I" + ascBase + "/asc",
        "-I" + ascBase + "/asc/include",
        "-I" + ascBase + "/asc/include/basic_api",
        "-I" + ascBase + "/asc/include/adv_api",
        "-I" + ascBase + "/asc/include/c_api",
        "-I" + ascBase + "/asc/impl/basic_api",
        "-I" + ascBase + "/asc/impl/adv_api",
        "-I" + ascBase + "/asc/impl/c_api",
        "-I" + ascBase + "/asc/impl/utils",
        "-include" + cannHome + "/include/version/asc_devkit_version.h",
    };
}

}  // namespace

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def(
        "rtc_handle",
        [](const std::string &src, const std::string &funcName) -> uint64_t {
            EnsureRt();
            aclrtcProg prog = nullptr;
            Check(aclrtcCreateProg(&prog, src.c_str(), funcName.c_str(), 0, nullptr, nullptr),
                  "aclrtcCreateProg");
            std::vector<std::string> optStrs = CompileOptions();
            std::vector<const char *> options;
            for (auto &o : optStrs) {
                options.push_back(o.c_str());
            }
            aclError e = aclrtcCompileProg(prog, static_cast<int>(options.size()), options.data());
            if (e != ACL_SUCCESS) {
                size_t logSize = 0;
                aclrtcGetCompileLogSize(prog, &logSize);
                std::string log(logSize, '\0');
                if (logSize > 0) {
                    aclrtcGetCompileLog(prog, &log[0]);
                }
                aclrtcDestroyProg(&prog);
                throw std::runtime_error("aclrtcCompileProg failed, log:\n" + log);
            }
            size_t binSize = 0;
            Check(aclrtcGetBinDataSize(prog, &binSize), "aclrtcGetBinDataSize");
            std::vector<char> bin(binSize);
            Check(aclrtcGetBinData(prog, bin.data()), "aclrtcGetBinData");
            aclrtcDestroyProg(&prog);
            aclrtBinaryLoadOption opt;
            opt.type = ACL_RT_BINARY_LOAD_OPT_MAGIC;
            opt.value.magic = ACL_RT_BINARY_MAGIC_ELF_AICORE;
            aclrtBinaryLoadOptions opts;
            opts.options = &opt;
            opts.numOpt = 1;
            aclrtBinHandle bh = nullptr;
            Check(aclrtBinaryLoadFromData(bin.data(), binSize, &opts, &bh),
                  "aclrtBinaryLoadFromData");
            aclrtFuncHandle fn = nullptr;
            Check(aclrtBinaryGetFunction(bh, funcName.c_str(), &fn), "aclrtBinaryGetFunction");
            return static_cast<uint64_t>(reinterpret_cast<uintptr_t>(fn));
        },
        py::arg("src"), py::arg("func_name"));

    // aclrtLaunchKernel takes a *packed* args buffer (the interposed call);
    // aclrtLaunchKernelWithArgsArray takes an array of per-argument pointers.
    m.def(
        "launch",
        [](uint64_t handle, uint32_t numBlocks, const std::string &blob, uint64_t stream) {
            EnsureRt();
            void *devArgs = nullptr;
            Check(aclrtMalloc(&devArgs, blob.size(), ACL_MEM_MALLOC_HUGE_FIRST),
                  "aclrtMalloc args");
            Check(aclrtMemcpy(devArgs, blob.size(), blob.data(), blob.size(),
                              ACL_MEMCPY_HOST_TO_DEVICE),
                  "aclrtMemcpy args");
            Check(aclrtLaunchKernel(reinterpret_cast<aclrtFuncHandle>(handle), numBlocks,
                                    devArgs, blob.size(),
                                    reinterpret_cast<aclrtStream>(stream)),
                  "aclrtLaunchKernel");
        },
        py::arg("handle"), py::arg("num_blocks"), py::arg("blob"), py::arg("stream"));
}
