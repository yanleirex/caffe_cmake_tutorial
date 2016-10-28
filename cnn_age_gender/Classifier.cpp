//
// Created by yanlei on 16-10-20.
//

#include "Classifier.h"

Classifier::Classifier(const string &model_file, const string &trained_file, const string &mean_file, const string& label_file)
{
    Caffe::set_mode(Caffe::CPU);

    /*
     * 载入模型
     */
    net_.reset(new Net<float>(model_file, TEST));
    net_->CopyTrainedLayersFrom(trained_file);
    Blob<float>* input_layer = net_->input_blobs()[0];
    num_channels_ = input_layer->channels();
    input_geometry_ = cv::Size(input_layer->width(), input_layer->height());

    setMean(mean_file);

    /*
     * 载入标签
     */
    std::ifstream labels(label_file.c_str());
    string line;
    while (std::getline(labels, line))
        labls_.push_back(string(line));

    Blob<float>* output_layer = net_->output_blobs()[0];
}


void Classifier::setMean(const string& mean_file)
{
    BlobProto blob_proto;
    ReadProtoFromBinaryFileOrDie(mean_file.c_str(), &blob_proto);

    /*
     * BlobProto 转换为 Blob
     */
    Blob<float> mean_blob;
    mean_blob.FromProto(blob_proto);
    CHECK_EQ(mean_blob.channels(), num_channels_)
        << "Number of channels of mean file doesn't match input layer.";
    std::vector<cv::Mat> channels;
    float* data = mean_blob.mutable_cpu_data();
    for(int i=0;i<num_channels_;++i)
    {
        cv::Mat channel(mean_blob.height(), mean_blob.width(), CV_32FC1, data);
        channels.push_back(channel);
        data += mean_blob.height()*mean_blob.width();
    }

    /*
     * 将单通道合并为一个图像
     */
    cv::Mat mean;
    cv::merge(channels, mean);

    /*
     * 计算全局均值
     */
    cv::Scalar channel_mean = cv::mean(mean);
    mean_ = cv::Mat(input_geometry_, mean.type(), channel_mean);
}


std::vector<Prediction> Classifier::Classify(cv::Mat& img, int N)
{
    std::vector<float> output = Predict(img);

    N = std::min<int>(labls_.size(), N);
    std::vector<int> maxN = Argmax(output, N);
    std::vector<Prediction> predictions;
    for(int i=0;i<N;++i)
    {
        int idx = maxN[i];
        predictions.push_back(std::make_pair(labls_[idx], output[idx]));
    }
    return predictions;
}

std::vector<float> Classifier::Predict(const cv::Mat& img)
{
    Blob<float>* input_layer = net_->input_blobs()[0];
    input_layer->Reshape(1, num_channels_, input_geometry_.height, input_geometry_.width);

    net_->Reshape();

    std::vector<cv::Mat> input_channels;

    WrapInputLayer(&input_channels);

    Preprocess(img, &input_channels);

    net_->Forward();

    /*
     * 输出
     */
    Blob<float>* output_layer = net_->output_blobs()[0];
    const float* begin = output_layer->cpu_data();
    const float* end = begin+output_layer->channels();
    /*
    float* predict = const_cast<float*>(output_layer->cpu_data()+output_layer->shape(2)*output_layer->shape(3));
    Eigen::Map<Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> prob(predict, output_layer->shape(2), output_layer->shape(3));
    */
    return std::vector<float>(begin, end);
}

void Classifier::WrapInputLayer(std::vector<cv::Mat> *input_channels)
{
    Blob<float>* input_layer = net_->input_blobs()[0];

    int width = input_layer->width();
    int height = input_layer->height();
    float* input_data = input_layer->mutable_cpu_data();
    for(int i=0;i<input_layer->channels();++i)
    {
        cv::Mat channel(height, width, CV_32FC1, input_data);
        input_channels->push_back(channel);
        input_data += width*height;
    }
}

void Classifier::Preprocess(const cv::Mat &img, std::vector<cv::Mat> *input_channels)
{
    cv::Mat sample;
    if(img.channels() == 3 && num_channels_==1)
    {
        cv::cvtColor(img, sample, cv::COLOR_BGR2GRAY);
    } else if(img.channels()==4 && num_channels_==1)
    {
        cv::cvtColor(img, sample, cv::COLOR_BGRA2GRAY);
    } else if(img.channels()==4 && num_channels_==3)
    {
        cv::cvtColor(img, sample, cv::COLOR_BGRA2BGR);
    } else if(img.channels()==1 && num_channels_ == 3)
    {
        cv::cvtColor(img, sample, cv::COLOR_GRAY2BGR);
    }
    else
    {
        sample = img;
    }

    cv::Mat sample_resized;
    if(sample.size() != input_geometry_)
    {
        cv::resize(sample, sample_resized, input_geometry_);
    }
    else
    {
        sample_resized = sample;
    }

    cv::Mat sample_float;
    if(num_channels_ == 3)
    {
        sample_resized.convertTo(sample_float, CV_32FC3);
    }
    else
    {
        sample_resized.convertTo(sample_float, CV_32FC1);
    }

    cv::Mat sample_normalized;
    cv::subtract(sample_float, mean_, sample_normalized);

    cv::split(sample_normalized, *input_channels);

    CHECK(reinterpret_cast<float*>(input_channels->at(0).data)
          == net_->input_blobs()[0]->cpu_data())
    << "Input channels are not wrapping the input layer of the network.";
}

