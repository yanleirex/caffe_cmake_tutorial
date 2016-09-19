//
// Created by yanlei on 16-9-19.
//
#include <caffe/caffe.hpp>
#include <caffe/util/io.hpp>
#include <caffe/blob.hpp>
#include <caffe/common.hpp>
#include <caffe/filler.hpp>

using namespace caffe;
using namespace std;

typedef double Dtype;

int main(int argc, char* argv[])
{
    Blob<Dtype>* blob_in = new Blob<Dtype>(20, 30, 40, 50);
    Blob<Dtype>* blob_out = new Blob<Dtype>(20, 30, 40, 50);
    int n = blob_in->count();

    //random number generation
    caffe_rng_uniform<Dtype>(n, -3, 3, blob_in->mutable_cpu_data());

    //sum
    Dtype asum;
    caffe_cpu_asum<Dtype>(n, blob_in->cpu_data());
    cout<<"asum: "<<asum<<endl;

    //sign signbit fasb, scale
    caffe_cpu_sign<Dtype>(n, blob_in->cpu_data(), blob_out->mutable_cpu_data());
    caffe_cpu_sgnbit<Dtype>(n, blob_in->cpu_data(), blob_out->mutable_cpu_data());
    //caffe_cpu_scale(n, 10, blob_in->cpu_data(), blob_out->mutable_cpu_data());

    const Dtype* x = blob_out->cpu_data();
    for(int i=0;i<n;i++)
        cout<<x[i]<<endl;

    delete blob_in;
    delete blob_out;
    return 0;
}

