//
// Created by yanlei on 16-10-19.
//

#ifndef CAFFE_CMAKE_DATA_TRANSFORMER_HPP
#define CAFFE_CMAKE_DATA_TRANSFORMER_HPP

#include <opencv2/opencv.hpp>

#include <eigen3/Eigen/Eigen>

#include <caffe/caffe.hpp>

const std::string project_root("/home/yanlei/Documents/code/cpp/ClionProjects/caffe_cmake/cnn_age_gender/");

Eigen::MatrixXf OpenCV2Eigen(const cv::Mat& image);

template <typename Dtype>
void Eigen2Blob(const std::vector<std::vector<Eigen::MatrixXf>> imgs, std::shared_ptr<caffe::Net<Dtype>> net)
{
    caffe::Blob<Dtype>* input_layer = net->input_blobs()[0];
    Dtype* input_data = input_layer->mutable_cpu_data();

    unsigned long img_number = imgs.size();
    unsigned long img_channel = imgs[0].size();
    unsigned long img_height = imgs[0][0].rows();
    unsigned long img_width = imgs[0][0].cols();

    unsigned long index = 0;

    for(int i=0;i<img_number;i++)
    {
        for(int c=0;c<img_channel;c++)
        {
            for(int h=0;h<img_height;h++)
            {
                for(int w=0;w<img_width;w++)
                {
                    *(input_data+index) = imgs[i][c](h, w);
                    index++;
                }
            }

        }
    }
}

#endif //CAFFE_CMAKE_DATA_TRANSFORMER_HPP
