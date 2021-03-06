#include <iostream>
#include <stdio.h>
#include <caffe/caffe.hpp>


using namespace caffe;
using namespace std;

typedef double Dtype;

int main() {
    std::cout << "Hello, World!" << std::endl;
    Blob<Dtype>* const blob = new Blob<Dtype>(20, 30, 40, 50);
    if(blob)
    {
        cout<<"Size of blob:";
        cout<<" N="<<blob->num();
        cout<<" K="<<blob->channels();
        cout<<" H="<<blob->height();
        cout<<" W="<<blob->width();
        cout<<" C="<<blob->count();
        cout<<endl;
    }

    //random sampling from uniform distribution
    FillerParameter fillerParameter;
    fillerParameter.set_min(-4);
    fillerParameter.set_max(5);
    UniformFiller<Dtype> filler(fillerParameter);
    filler.Fill(blob);

    //absolute sum of blob
    Dtype expected_asum = 0;
    const Dtype* data = blob->cpu_data();
    for(int i=0;i<blob->count();++i)
    {
        expected_asum += fabs(data[i]);
    }
    cout<<"expected asum of blob:"<<expected_asum<<endl;
    cout<<"asum of blob on cpu:"<<blob->asum_data()<<endl;

    delete blob;
    return 0;
}