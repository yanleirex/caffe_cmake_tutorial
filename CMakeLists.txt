cmake_minimum_required(VERSION 3.6)
project(caffe_cmake)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/bin)

find_package(Caffe)
include_directories(${Caffe_INCLUDE_DIRS})
add_definitions(${Caffe_DEFINITIONS})

find_package(OpenCV REQUIRED)


include_directories(/usr/local/include)

add_executable(caffe_cmake main.cpp)
target_link_libraries(caffe_cmake ${Caffe_LIBRARIES})

add_executable(ex_blob ex_blob.cpp)
target_link_libraries(ex_blob ${Caffe_LIBRARIES})

add_executable(ex_math ex_math.cpp)
target_link_libraries(ex_math ${Caffe_LIBRARIES})

add_executable(ex_dataset ex_dataset.cpp)
target_link_libraries(ex_dataset ${Caffe_LIBRARIES})

add_executable(ex_layer ex_layer.cpp)
target_link_libraries(ex_layer ${Caffe_LIBRARIES})

add_executable(bindb bindb.cpp)
target_link_libraries(bindb ${Caffe_LIBRARIES})

add_executable(logreg_train logreg_train.cpp)
target_link_libraries(logreg_train ${Caffe_LIBRARIES})

add_executable(logreg_test logreg_test.cpp)
target_link_libraries(logreg_test ${Caffe_LIBRARIES})

add_executable(test_caffe_manual test_caffe_manual.cpp)
target_link_libraries(test_caffe_manual ${Caffe_LIBRARIES})

add_executable(readMnistImage readMnistImage.cpp)
target_link_libraries(readMnistImage ${OpenCV_LIBS})

add_executable(testTrainedMnist testTrainedMnist.cpp)
target_link_libraries(testTrainedMnist ${OpenCV_LIBS})
target_link_libraries(testTrainedMnist ${Caffe_LIBRARIES})

add_subdirectory(caffe_examples)

add_subdirectory(cnn_age_gender)

add_subdirectory(AgeAndGenderEstimation-master)