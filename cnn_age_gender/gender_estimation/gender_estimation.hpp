//
// Created by yanlei on 16-10-20.
//

#ifndef CAFFE_CMAKE_GENDER_ESTIMATION_HPP
#define CAFFE_CMAKE_GENDER_ESTIMATION_HPP

#include <vector>
#include <string>

#include <opencv2/opencv.hpp>

#include "../utils/tool_box.hpp"

enum class Gender{Man, Woman, Error};

static std::map<Gender, std::string> gender_lists
        {
                {Gender ::Man, "Man"},
                {Gender::Woman, "Woman"},
                {Gender::Error, "Error"}
        };

std::vector<Gender> gender_estimation(const cv::Mat&, const std::vector<BoundingBox>&);
#endif //CAFFE_CMAKE_GENDER_ESTIMATION_HPP
