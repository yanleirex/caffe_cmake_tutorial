file(GLOB_RECURSE srcs ${CMAKE_CURRENT_SOURCE_DIR}/*/*.cpp)
file(GLOB_RECURSE hpps ${CMAKE_CURRENT_SOURCE_DIR}/*/*.hpp)

add_executable(age_gender main.cpp ${srcs} ${hpps})
target_link_libraries(age_gender ${Caffe_LIBRARIES} ${OpenCV_LIBS})
