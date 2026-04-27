cmake_minimum_required(VERSION 3.25)

include(FetchContent)

FetchContent_Declare(
        googletest
        GIT_REPOSITORY https://github.com/google/googletest.git
        GIT_TAG v1.14.0
)

set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(googletest)

include(GoogleTest)

add_executable(autoresearchpp_test
        test_Config.cpp
        test_Device.cpp
        test_Tokenizer.cpp
        test_Dataset.cpp
        test_Model.cpp
        test_Evaluator.cpp
)

target_link_libraries(autoresearchpp_test
        PRIVATE
        autoresearchpp_lib
        GTest::gtest_main
)

target_link_options(autoresearchpp_test PRIVATE
        -Wl,--allow-shlib-undefined
)

target_compile_options(autoresearchpp_test PRIVATE
        -Wall
        -Wextra
        -Wpedantic
)

gtest_discover_tests(autoresearchpp_test)