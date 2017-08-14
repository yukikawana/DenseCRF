cmake_minimum_required (VERSION 2.6)
project(densecrf)

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} ${PROJECT_SOURCE_DIR}/cmake)
#set(CMAKE_CXX_FLAGS_DEBUG "-g3 -O0 -pg")
include_directories(include)

add_subdirectory (src)
add_subdirectory (examples)
add_subdirectory (external)


