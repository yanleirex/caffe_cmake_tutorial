//
// Created by yanlei on 16-10-19.
//

#ifndef CAFFE_CMAKE_FACE_DETECTION_HPP
#define CAFFE_CMAKE_FACE_DETECTION_HPP

#include <string>
#include <vector>
#include <fstream>
#include <eigen3/Eigen/Eigen>
#include <opencv2/opencv.hpp>

#include "../utils/data_transformer.hpp"
#include "../utils/BoundingBox.hpp"

const float red_channel_mean = 104.00698793;
const float green_channel_mean = 116.66876762;
const float blue_channel_mean = 122.67891434;

void face_detecton(const cv::Mat&, std::vector<BoundingBox>&);

void modify_prototxt(const int, const int);

void add_to_boundingbox(const Eigen::MatrixXf&, const float, std::vector<BoundingBox>&);

#endif //CAFFE_CMAKE_FACE_DETECTION_HPP
