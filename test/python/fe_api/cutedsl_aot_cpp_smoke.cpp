#include <tvm/ffi/extra/module.h>
#include <tvm/ffi/function.h>

#include <exception>
#include <iostream>
#include <string>

int main(int argc, char** argv) {
  if (argc != 3) {
    std::cerr << "usage: cutedsl_aot_cpp_smoke <shared-library> <symbol>\n";
    return 2;
  }

  const std::string shared_library = argv[1];
  const std::string symbol = argv[2];

  try {
    tvm::ffi::Module module = tvm::ffi::Module::LoadFromFile(shared_library);
    auto function = module->GetFunction(symbol);
    if (!function.has_value()) {
      function = module->GetFunction("__tvm_ffi_" + symbol);
    }
    if (!function.has_value()) {
      std::cerr << "TVM-FFI module did not expose function " << symbol << "\n";
      return 3;
    }

    try {
      function.value()();
    } catch (const std::exception& err) {
      std::cout << "TVM-FFI function invocation reached expected argument validation: "
                << err.what() << "\n";
      return 0;
    }
  } catch (const std::exception& err) {
    std::cerr << "TVM-FFI module smoke failed: " << err.what() << "\n";
    return 4;
  }

  return 0;
}
