#include <caffe/caffe.hpp>
#include <caffe/blob.hpp>
#include <caffe/common.hpp>
#include <caffe/filler.hpp>

using namespace caffe;
using namespace std;
using namespace cv;

typedef double Dtype;

clock_t tStart, tEnd;
#define COMPTIME(X) \
cout<<"CompTime of "<<(X)<<": "<<(double)(tEnd-tStart)/CLOCKS_PER_SEC<<endl;

int main(int argc, char* argv[])
{
    //Initialization
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
    fillerParameter.set_min(-3);
    fillerParameter.set_max(3);
    UniformFiller<Dtype> filler(fillerParameter);
    filler.Fill(blob);

    //sum of squares
    //access data on the host
    Dtype expected_sumq = 0;
    const Dtype* data = blob->cpu_data();
    for(int i=0;i<blob->count();++i)
    {
        expected_sumq += data[i]*data[i];
    }
    cout<<endl;
    cout<<"expected sumsq of blob:"<<expected_sumq<<endl;
    tStart = clock();
    cout<<"sumq of blob on cpu:"<<blob->sumsq_data()<<endl;
    tEnd = clock();
    COMPTIME("sumsq of blob on cpu");
}