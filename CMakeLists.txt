cmake_minimum_required(VERSION 2.8)
project(FrameCompare C CXX)

# Main settings
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/lib)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/lib)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/bin)

# OpenCV dependency
find_package(OpenCV REQUIRED)
if(MSVC)
	add_definitions(-D_CRT_SECURE_NO_WARNINGS)
endif()

# Project settings
set(frcmp_target "frcmp")
file(GLOB frcmp_files RELATIVE ${CMAKE_SOURCE_DIR} ${CMAKE_SOURCE_DIR}/src/*.cpp ${CMAKE_SOURCE_DIR}/src/*.h)
message("frcmp_files = '${frcmp_files}'")
add_executable(${frcmp_target} ${frcmp_files})
target_link_libraries(${frcmp_target} ${OpenCV_LIBS})
