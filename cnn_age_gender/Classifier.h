//
// Created by yanlei on 16-10-20.
//

#ifndef CAFFE_CMAKE_CLASSIFIER_H
#define CAFFE_CMAKE_CLASSIFIER_H

#include <string>
#include <vector>
#include <algorithm>
#include <memory>
#include <utility>
#include <iosfwd>
#include <opencv2/opencv.hpp>
#include <eigen3/Eigen/Eigen>

#include <caffe/caffe.hpp>

using namespace caffe;

static bool PairCompare(const std::pair<float, int>& lhs, const std::pair<float, int>& rhs);
static std::vector<int> Argmax(const std::vector<float>& v, int N);

static bool PairCompare(const std::pair<float, int>& lhs, const std::pair<float, int>& rhs)
{
    return lhs.first>rhs.first;
}

static std::vector<int> Argmax(const std::vector<float>& v, int N)
{
    std::vector<std::pair<float, int> > pairs;
    for(size_t i=0;i<v.size();++i)
    {
        pairs.push_back(std::make_pair(v[i], i));
    }
    std::partial_sort(pairs.begin(), pairs.begin()+N, pairs.end(), PairCompare);

    std::vector<int> result;
    for(int i=0;i<N;++i)
        result.push_back(pairs[i].second);
    return result;
}

typedef std::pair<string, float> Prediction;

class Classifier {
public:
    Classifier(const string& model_file,
            const string& trained_file,
            const string& mean_file,
            const string& label_file);

    std::vector<Prediction> Classify(cv::Mat& img, int N);


private:
    void setMean(const string& mean_file);
    std::vector<float> Predict(const cv::Mat& img);

    void WrapInputLayer(std::vector<cv::Mat>* input_channels);

    void Preprocess(const cv::Mat& img, std::vector<cv::Mat>* input_channels);

private:
    std::shared_ptr<caffe::Net<float> > net_;
    cv::Size input_geometry_;
    int num_channels_;
    cv::Mat mean_;
    std::vector<string> labls_;

};


#endif //CAFFE_CMAKE_CLASSIFIER_H
