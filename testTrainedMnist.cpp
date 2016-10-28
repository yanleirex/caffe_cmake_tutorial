//
// Created by yanlei on 16-10-28.
//
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <string>

#include <caffe/util/db.hpp>
#include <caffe/util/format.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>

void WriteToDb(const cv::Mat& image, const int& label, const std::string dbPath)
{
    std::shared_ptr<caffe::db::DB> db(caffe::db::GetDB("lmdb"));
    db->Open(dbPath, caffe::db::NEW);
    std::shared_ptr<caffe::db::Transaction> txn(db->NewTransaction());

    std::string value;
    caffe::Datum datum;
    datum.set_channels(1);
    datum.set_height(image.rows);
    datum.set_width(image.cols);
    datum.set_data(image.data, image.rows*image.cols);
    datum.set_label(label);
    std::string keyStr = caffe::format_int(0, 0);
    datum.SerializeToString(&value);
    txn->Put(keyStr, value);
    txn->Commit();
    db->Close();
}

int main(int argc, char** argv)
{
    if(argc!=4)
    {
        std::cout<<"Usage:TestTrainedMnist <Test image><label><output folder>"<<std::endl;
        exit(EXIT_FAILURE);
    }
    for(int i=0;i<argc;i++)
    {
        std::cout<<"arg"<<i<<":"<<argv[i]<<std::endl;
    }
    cv::Mat testImg = cv::imread(argv[1], CV_LOAD_IMAGE_GRAYSCALE);

    cv::bitwise_not(testImg, testImg);
    if(testImg.empty())
    {
        std::cout<<"Fail to read test image"<<std::endl;
        exit(EXIT_FAILURE);
    }

    const int& label = std::atoi(argv[2]);
    std::cout<<"label:"<<label<<std::endl;
    WriteToDb(testImg, label, argv[3]);
    cv::imshow("testImg", testImg);
    cv::waitKey();
    exit(EXIT_SUCCESS);
}
