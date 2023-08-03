cmake_minimum_required(VERSION 2.8.12)

project(Test)

#set(CMAKE_INCLUDE_CURRENT_DIR ON)
#set(CMAKE_AUTOMOC ON)
set(CMAKE_CXX_FLAGS "-std=c++11 -g -Wall")

set(PYTHON_INCLUDE_DIRS "/home/ylr/anaconda3/include/python3.7m/")
INCLUDE_DIRECTORIES(${PYTHON_INCLUDE_DIRS})
link_directories(/home/ylr/anaconda3/lib/python3.7/config-3.7m-x86_64-linux-gnu)
set(PYTHON_LIBRARIES "/home/ylr/anaconda3/lib/libpython3.7m.so")

find_package(OpenCV 3.2.0 REQUIRED)
add_executable(Test "main.cpp")
target_link_libraries(Test ${PYTHON_LIBRARIES} ${OpenCV_LIBS})
