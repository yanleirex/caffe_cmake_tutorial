set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/bin)

file(GLOB_RECURSE srcs ${CMAKE_CURRENT_SOURCE_DIR}/*/*.cpp)
file(GLOB_RECURSE hpps ${CMAKE_CURRENT_SOURCE_DIR}/*/*.hpp)

add_executable(cnn_age_gender  main.cpp ${srcs} ${hpps})
target_link_libraries(cnn_age_gender ${Caffe_LIBRARIES} ${OpenCV_LIBS})

add_executable(age_gender_estimator age_gender_estimator.cpp Classifier.h Classifier.cpp)
target_link_libraries(age_gender_estimator ${Caffe_LIBRARIES} ${OpenCV_LIBS})