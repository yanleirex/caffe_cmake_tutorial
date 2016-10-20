//
// Created by yanlei on 16-10-19.
//

#ifndef CAFFE_CMAKE_TOOL_BOX_HPP
#define CAFFE_CMAKE_TOOL_BOX_HPP

#include <caffe/caffe.hpp>
#include <map>

#include "BoundingBox.hpp"
#include "data_transformer.hpp"

std::vector<float> SetMean(const std::string&, int);

BoundingBox extend_face_to_whole_head(const BoundingBox&, const int, const int);

const int cellsize = 227;

std::vector<int> Argmax(const std::vector<float>& v, int N);


#endif //CAFFE_CMAKE_TOOL_BOX_HPP
