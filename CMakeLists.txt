cmake_minimum_required(VERSION 3.20)
project(onpairplus CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if (NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif ()

add_library(onpairplus
        src/OnPairPlus.cpp
        src/opp/OnPairPlusEncoder.cpp
        src/opp/OnPairPlusDecoder.cpp
        src/opp/Tokenizer.cpp)
target_include_directories(onpairplus PUBLIC src thirdparty)

# The training path is hash-lookup bound. Boost's flat map is the fastest we measured (dbtext
# encoding: 96 MiB/s vs 79 for the vendored unordered_dense and 64 for std::unordered_map), so use
# it when its headers are present and fall back to the vendored header otherwise. Both produce
# byte-identical output. Set OPP_USE_BOOST=OFF to force the vendored map.
option(OPP_USE_BOOST "Use boost::unordered_flat_map for training when available" ON)
if (OPP_USE_BOOST)
    find_path(OPP_BOOST_INCLUDE_DIR boost/unordered/unordered_flat_map.hpp
            HINTS /opt/homebrew/include /usr/local/include /usr/include)
    if (OPP_BOOST_INCLUDE_DIR)
        message(STATUS "onpairplus: using boost::unordered_flat_map (${OPP_BOOST_INCLUDE_DIR})")
        target_include_directories(onpairplus PUBLIC "${OPP_BOOST_INCLUDE_DIR}")
        target_compile_definitions(onpairplus PUBLIC OPP_BOOST_UNORDERED)
    else ()
        message(STATUS "onpairplus: boost not found, using vendored ankerl::unordered_dense")
    endif ()
endif ()

add_executable(opp tools/opp.cpp)
target_link_libraries(opp PRIVATE onpairplus)

# Tests (only when building this project directly, not as a subdirectory of the benchmark)
if (PROJECT_IS_TOP_LEVEL)
    enable_testing()
    include("${CMAKE_CURRENT_SOURCE_DIR}/vendor/catch2.cmake")
    add_executable(tests tests/Tests.cpp)
    target_link_libraries(tests PRIVATE onpairplus Catch2::Catch2WithMain)
    include(Catch)
    catch_discover_tests(tests)
endif ()
