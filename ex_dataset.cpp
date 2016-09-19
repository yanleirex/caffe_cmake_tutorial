//
// Created by yanlei on 16-9-19.
//
#include <cstring>
#include <cstdlib>
#include <vector>

#include <string>
#include <iostream>
#include <stdio.h>

#include <opencv2/highgui.hpp>
#include <caffe/caffe.hpp>
#include <caffe/util/io.hpp>
#include <caffe/blob.hpp>
#include <caffe/common.hpp>
#include <caffe/filler.hpp>

using namespace caffe;
using namespace std;
using namespace cv;

typedef double Dtype;

uint32_t  swap_ending(uint32_t val)
{
    val = ((val<<8)&0xFF00FF00)|((val>>8)&0xFF00FF);
    return (val<<16) | (val>>16);
}

int main(int argc, char** argv)
{
    const char* image_filename = "/home/yanlei/caffe/data/mnist/train-images-idx3-ubyte";
    const char* label_filename = "/home/yanlei/caffe/data/mnist/train-labels-idx1-ubyte";

    //open files
    std::ifstream image_file(image_filename, std::ios::in|std::ios::binary);
    std::ifstream label_file(label_filename, std::ios::in|std::ios::binary);

    CHECK(image_file)<<"Unable to open file "<<image_filename;
    CHECK(label_file)<<"Unable to open file "<<label_filename;

    //Read the magic and the meta data
    uint32_t magic;
    uint32_t num_items;
    uint32_t num_labels;
    uint32_t rows;
    uint32_t cols;

    image_file.read(reinterpret_cast<char*>(&magic), 4);
    magic = swap_ending(magic);
    CHECK_EQ(magic, 2051)<<"Incorrect image file magic";
    label_file.read(reinterpret_cast<char*>(&magic), 4);
    magic = swap_ending(magic);
    CHECK_EQ(magic, 2049)<<"Incorrect label file magic";

    image_file.read(reinterpret_cast<char*>(&num_items), 4);
    num_items = swap_ending(num_items);
    label_file.read(reinterpret_cast<char*>(&num_labels), 4);
    num_labels = swap_ending(num_labels);
    CHECK_EQ(num_items, num_labels);
    image_file.read(reinterpret_cast<char*>(&rows), 4);
    rows = swap_ending(rows);
    image_file.read(reinterpret_cast<char*>(&cols), 4);
    cols = swap_ending(cols);

    cout<<"A total of "<<num_items<<" items"<<endl;
    cout<<"Rows: "<<rows<<" Cols:"<<cols<<endl;
    char label;
    char* pixels = new char[rows*cols];

    for(int item_id = 0;item_id<num_items;++item_id)
    {
        image_file.read(pixels, rows*cols);
//        for(int j=0;j<rows*cols;j++)
//        {
//            cout<<pixels;
//        }
//        cout<<endl;
        label_file.read(&label, 1);
    }

    delete pixels;

    return 0;
}
