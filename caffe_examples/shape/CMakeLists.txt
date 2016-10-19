file(GLOB_RECURSE srcs ${CMAKE_CURRENT_SOURCE_DIR}/*.cpp)

#build each source file independently
foreach(source ${srcs})
    get_filename_component(name ${source} NAME_WE)

    add_executable(${name} ${source})
    target_link_libraries(${name} ${Caffe_LIBRARIES} ${OpenCV_LIBS})
endforeach(source)