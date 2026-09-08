cmake_minimum_required(VERSION 3.20)

# Self-contained benchmark project: vcpkg and the upstream FSST sources are bootstrapped into the
# build directory; nothing outside benchmark/ depends on any of this.

# Bootstrap vcpkg (provides lz4, zstd, brotli via vcpkg.json manifest mode)
set(VCPKG_ROOT "${CMAKE_CURRENT_BINARY_DIR}/vcpkg" CACHE PATH "")
if (NOT EXISTS "${VCPKG_ROOT}")
    message(STATUS "Cloning vcpkg...")
    execute_process(COMMAND git clone --depth 1 https://github.com/microsoft/vcpkg.git "${VCPKG_ROOT}"
            COMMAND_ERROR_IS_FATAL ANY)
    execute_process(COMMAND "${VCPKG_ROOT}/bootstrap-vcpkg.sh" -disableMetrics
            WORKING_DIRECTORY "${VCPKG_ROOT}" COMMAND_ERROR_IS_FATAL ANY)
endif ()
set(CMAKE_TOOLCHAIN_FILE "${VCPKG_ROOT}/scripts/buildsystems/vcpkg.cmake" CACHE STRING "")

project(opp-benchmark CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if (NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif ()

find_package(lz4 CONFIG REQUIRED)
find_package(zstd CONFIG REQUIRED)
find_package(unofficial-brotli CONFIG REQUIRED)

# Upstream FSST is not packaged by vcpkg; fetch and build it directly
set(FSST_SRC "${CMAKE_CURRENT_BINARY_DIR}/fsst-src")
if (NOT EXISTS "${FSST_SRC}")
    message(STATUS "Cloning fsst...")
    execute_process(COMMAND git clone --depth 1 https://github.com/cwida/fsst "${FSST_SRC}"
            COMMAND_ERROR_IS_FATAL ANY)
endif ()
add_library(fsst_upstream STATIC "${FSST_SRC}/libfsst.cpp" "${FSST_SRC}/fsst_avx512.cpp")
target_include_directories(fsst_upstream PUBLIC "${FSST_SRC}")
target_compile_options(fsst_upstream PRIVATE -w) # upstream code, silence its warnings

# The SGTT tokenizer baseline from token-vldb2026. Its token dictionary is embedded via C23
# #embed there, which Apple clang lacks - tokenizer_gen.py emits the arrays as hex literals.
set(TOKEN_VLDB_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../../token-vldb2026" CACHE PATH
        "Checkout of the token-vldb2026 repository (for the tokenizer baseline)")
set(TOKENS_DIR "${TOKEN_VLDB_DIR}/src/tokens/openai100kpatched")
set(TOKEN_DATA_CPP "${CMAKE_CURRENT_BINARY_DIR}/openai100kpatched_data.cpp")
add_custom_command(OUTPUT "${TOKEN_DATA_CPP}"
        COMMAND python3 "${CMAKE_CURRENT_SOURCE_DIR}/tokenizer_gen.py" "${TOKENS_DIR}" "${TOKEN_DATA_CPP}"
        DEPENDS tokenizer_gen.py "${TOKENS_DIR}/openai100kpatched_dict.bin" "${TOKENS_DIR}/openai100kpatched_lengths.bin")
add_library(sgtt_tokenizer STATIC
        "${TOKEN_VLDB_DIR}/src/compressor/TokenizerCompressor.cpp"
        "${TOKEN_DATA_CPP}"
        tokenizer_stubs.cpp)
target_include_directories(sgtt_tokenizer PUBLIC "${TOKEN_VLDB_DIR}/src")
# reference code, silence its warnings and patch in std headers it gets transitively upstream
target_compile_options(sgtt_tokenizer PRIVATE -w -include "${CMAKE_CURRENT_SOURCE_DIR}/tokenizer_prelude.hpp")

# The reference OnPair implementation (vendored in token-vldb2026/thirdparty), i.e. the algorithm
# OnPair+ extends - built straight from its sources so it is measured on the same footing
set(ONPAIR_DIR "${TOKEN_VLDB_DIR}/thirdparty/onpair")
add_library(onpair_upstream STATIC "${ONPAIR_DIR}/src/onpair.cpp" "${ONPAIR_DIR}/src/onpair16.cpp")
target_include_directories(onpair_upstream PUBLIC "${ONPAIR_DIR}/include" "${ONPAIR_DIR}/external/robin_hood")
target_compile_options(onpair_upstream PRIVATE -w) # upstream code, silence its warnings

# The compressor under test
add_subdirectory("${CMAKE_CURRENT_SOURCE_DIR}/.." "${CMAKE_CURRENT_BINARY_DIR}/onpairplus")

add_executable(benchmark main.cpp)
target_link_libraries(benchmark PRIVATE
        onpairplus
        sgtt_tokenizer
        onpair_upstream
        lz4::lz4
        zstd::libzstd
        unofficial::brotli::brotlienc
        unofficial::brotli::brotlidec
        fsst_upstream)
