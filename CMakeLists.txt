cmake_minimum_required(VERSION 3.16)
project(mission_interpreter CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

include(FetchContent)

set(LLAMA_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_TESTS    OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_TOOLS    OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_SERVER   OFF CACHE BOOL "" FORCE)
set(LLAMA_CURL           OFF CACHE BOOL "" FORCE)
add_subdirectory(llama.cpp)


set(YAML_CPP_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(YAML_CPP_BUILD_TOOLS OFF CACHE BOOL "" FORCE)

FetchContent_Declare(yaml-cpp
    GIT_REPOSITORY https://github.com/jbeder/yaml-cpp.git
    GIT_TAG        yaml-cpp-0.9.0)
FetchContent_MakeAvailable(yaml-cpp)

add_library(mission_interpreter src/mission_interpreter.cpp)
target_include_directories(mission_interpreter
    PUBLIC  include
)
target_link_libraries(mission_interpreter
    PUBLIC  llama
    PRIVATE yaml-cpp::yaml-cpp)

add_executable(mission_cli src/main.cpp)
target_link_libraries(mission_cli PRIVATE mission_interpreter yaml-cpp::yaml-cpp)
