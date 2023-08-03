cmake_minimum_required(VERSION 2.8.12)

project(Gesture_Object_R)
set(CMAKE_CXX_FLAGS "-std=c++11 -g -Wall")
set(CMAKE_PREFIX_PATH ${CMAKE_PREFIX_PATH} $ENV{HOME}/freenect2/lib/cmake/freenect2)

set(PYTHON_INCLUDE_DIRS "/home/ylr/anaconda3/include/python3.7m/")
INCLUDE_DIRECTORIES(${PYTHON_INCLUDE_DIRS})
link_directories(/home/ylr/anaconda3/lib/python3.7/config-3.7m-x86_64-linux-gnu)
set(PYTHON_LIBRARIES "/home/ylr/anaconda3/lib/libpython3.7m.so")

find_package(OpenCV 3.2.0 REQUIRED)
find_package(OpenGL REQUIRED)
find_package(freenect2 REQUIRED)
find_package(PCL 1.8.1 REQUIRED)
include_directories(${OPENGL_INCLUDE_DIR})
include_directories("/usr/include/libusb-1.0/")
include_directories(/home/ylr/libfreenect2/include)
include_directories(${PCL_INCLUDE_DIRS})
INCLUDE_DIRECTORIES(
  ${freenect2_INCLUDE_DIR}
  ${PCL_INCLUDE_DIRS}
)
link_directories(
  ${PCL_LIBRARY_DIRS}
)
add_definitions(
  ${PCL_DEFINITIONS}
)

find_package(GLUT REQUIRED)
include_directories(${GLUT_INCLUDE_DIR})
include_directories(/home/ylr/LeapDeveloperKit_2.3.1+31549_linux/LeapSDK/include)#头文件路径
link_directories(/home/ylr/LeapDeveloperKit_2.3.1+31549_linux/LeapSDK/lib/x64)#lib路径

add_executable(Gesture_Object_R "main.cpp")

target_link_libraries(Gesture_Object_R ${PYTHON_LIBRARIES})
target_link_libraries(Gesture_Object_R ${GLUT_LIBRARY} ${OPENGL_LIBRARY})
target_link_libraries(Gesture_Object_R /home/ylr/LeapDeveloperKit_2.3.1+31549_linux/LeapSDK/lib/x64/libLeap.so)# .so路径//
target_link_libraries(Gesture_Object_R ${freenect2_LIBRARIES} ${OpenCV_LIBS} ${PCL_LIBRARIES} ${LEAP_LINK_LIBRARY})
