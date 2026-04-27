cmake_minimum_required(VERSION 3.25)
project(autoresearch++ VERSION 1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

find_package(Torch REQUIRED)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")

file(GLOB_RECURSE HEADERS
		${CMAKE_CURRENT_SOURCE_DIR}/include/*.h
		${CMAKE_CURRENT_SOURCE_DIR}/include/*.hpp
)
message(STATUS "Found headers: ${HEADERS}")

add_library(autoresearchpp_lib
		src/Config.cpp
		src/Device.cpp
		src/Tokenizer.cpp
		src/Dataset.cpp
		src/Model.cpp
		src/Optimizer.cpp
		src/Trainer.cpp
		src/Evaluator.cpp
		src/ExperimentLog.cpp
		src/CliParser.cpp
		src/Metricslog.cpp
		${HEADERS}
)

target_include_directories(autoresearchpp_lib
		PUBLIC
		$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
		$<INSTALL_INTERFACE:include>
)

target_link_libraries(autoresearchpp_lib PUBLIC "${TORCH_LIBRARIES}")

target_compile_options(autoresearchpp_lib PRIVATE
		-Wall
		-Wextra
		-Wpedantic
)

add_executable(autoresearchpp src/main.cpp)
target_link_options(autoresearchpp PRIVATE
		-Wl,--allow-shlib-undefined
)
target_link_libraries(autoresearchpp PRIVATE autoresearchpp_lib)

option(BUILD_TESTS "Build unit tests" OFF) # was on
if(BUILD_TESTS)
	enable_testing()
	add_subdirectory(test)
endif()