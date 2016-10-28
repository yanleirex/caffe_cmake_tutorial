//
// Created by yanlei on 16-10-28.
//
#include <fstream>
#include <iostream>
#include <vector>

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/stitching/stitcher.hpp>

bool ReadMnistInt(std::ifstream& ifs, int& i)
{
    if(!ifs.is_open())
    {
        std::cout<<"Error: ifstream is not opened"<<std::endl;
        return false;
    }
    char char4[4] = {0};
    ifs.read(char4, 4);

    //swap char4
    char temp = char4[0];
    char4[0] = char4[3];
    char4[3] = temp;
    temp = char4[1];
    char4[1] = char4[2];
    char4[2] = temp;

    i = *reinterpret_cast<int*>(char4);

    return true;
}

cv::Mat ReadMnistImage(std::ifstream& ifs, const size_t& width, const size_t& height)
{
    if(!ifs.is_open())
    {
        std::cout<<"Error: ifstream is not opened"<<std::endl;
        return cv::Mat();
    }
    char pixel;
    cv::Mat mat(height, width, CV_8UC1);
    for(size_t v=0;v<height;++v)
    {
        for(size_t u=0;u<width;++u)
        {
            ifs.read(&pixel, 1);
            pixel ^= 0xff;
            mat.at<unsigned char>(v, u) = static_cast<unsigned char>(pixel);
        }
    }
    return mat;
}

int main(int argc, char** argv)
{
    if(argc!=2)
    {
        std::cout<<"Usage:ReadMnistImage <MinistImages>"<<std::endl;
        exit(EXIT_FAILURE);
    }
    int magicNumber = 0;
    int numImage = 0;
    int width = 0;
    int height = 0;
    std::ifstream ifs(argv[1], std::ios::in | std::ios::binary);
    ReadMnistInt(ifs, magicNumber);
    ReadMnistInt(ifs, numImage);
    ReadMnistInt(ifs, width);
    ReadMnistInt(ifs, height);

    if(numImage <= 0 || width <=0 || height<=0)
    {
        std::cout<<"Error:numImage("<<numImage<<"),"
                                              <<"width("<<width<<"),"
                                                <<"height("<<height<<"),"
                                                <<std::endl;
        exit(EXIT_FAILURE);
    }
    std::cout<<"magicNumber:"<<magicNumber<<std::endl;
    std::cout<<"numImage:"<<numImage<<std::endl;
    std::cout<<"width:"<<width<<std::endl;
    std::cout<<"height:"<<height<<std::endl;
    for(unsigned int i=0;i<numImage/100;++i)
    {
        cv::Mat imageShow(width*10, height*10, CV_8UC1);
        //cv::Mat imageShow;
        for(int i_index = 0;i_index<10;i_index++)
        {
            for(int j_index=0;j_index<10;j_index++)
            {
                //cv::Mat img = ReadMnistImage(ifs, width, height);
                for(int h=0;h<height;h++)
                {
                    for(int w=0;w<width;w++)
                    {
                        char pixel;
                        ifs.read(&pixel, 1);
                        pixel ^= 0xff;
                        imageShow.at<unsigned char>(i_index*28+h, j_index*28 + w) = static_cast<unsigned char>(pixel);
                    }
                }
            }
        }
        cv::imshow("img", imageShow);
        if(27==cv::waitKey())
            break;
    }
    exit(EXIT_SUCCESS);
}

