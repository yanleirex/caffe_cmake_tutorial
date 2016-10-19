//
// Created by yanlei on 16-10-19.
//
#include <iostream>
#include <string>
#include <vector>

#include <caffe/caffe.hpp>

using namespace caffe;

/*
 * 根据layer名称，获取layer索引号
 */
unsigned int get_layer_index(boost::shared_ptr<Net<float> >&net, char *query_layer_name)
{
    std::string str_query(query_layer_name);
    vector<string> const& layer_names = net->layer_names();
    for(unsigned int i=0;i!=layer_names.size();++i)
    {
        if(str_query==layer_names[i])
        {
            return i;
        }
    }
    LOG(FATAL)<<"Unknown layer name: "<<str_query;
}

/*
 * 根据blob名字获取blob索引
 */
unsigned int get_blob_index(boost::shared_ptr<Net<float> >& net, char *query_blob_name)
{
    std::string str_query(query_blob_name);
    vector<string> const& blob_names = net->blob_names();
    for(unsigned int i=0;i!=blob_names.size();++i)
    {
        if(str_query==blob_names[i])
        {
            return i;
        }
    }
    LOG(FATAL)<<"Unknown blob name: "<<str_query;
}

void get_blob_features(boost::shared_ptr<Net<float> >& net, float* data_ptr, char* layer_name)
{
    unsigned int id = get_layer_index(net, layer_name);
    const vector<Blob<float>*>& output_blobs = net->top_vecs()[id];
    for(unsigned int i=0;i<output_blobs.size();++i)
    {
        switch (Caffe::mode())
        {
            case Caffe::CPU:
                memcpy(data_ptr, output_blobs[i]->cpu_data(), sizeof(float)*output_blobs[i]->count());
                break;
            case Caffe::GPU:
                LOG(FATAL)<<"GPU mode not support.";
                break;
            default:
                LOG(FATAL)<<"Unknown Caffe mode.";
        }
    }
}

int main(int argc, char** argv)
{
    char *proto = "../cnn_age_gender/deploy_age.prototxt";
    char *model = "../cnn_age_gender/age_net.caffemodel";
    char *mean_file = "../cnn_age_gender/mean.binaryproto";

    Phase phase = TEST;

    Caffe::set_mode(Caffe::CPU);
    //Caffe::SetDevice(0);

    //Process mean
    Blob<float> image_mean;
    BlobProto blob_proto;
    const float *mean_ptr;
    bool succeed = ReadProtoFromBinaryFile(mean_file, &blob_proto);
    if(succeed)
    {
        std::cout<<"read image mean succeeded"<<std::endl;
        image_mean.FromProto(blob_proto);
        mean_ptr = (const float *)image_mean.cpu_data();
        unsigned int num_pixel = image_mean.count();
        std::cout<<num_pixel<<"\n";
        std::cout<<image_mean.num()<<"\t"<<image_mean.channels()<<"\t"<<image_mean.height()<<"\t"<<image_mean.width()<<"\n";
        std::cout<<mean_ptr[0]<<"\t"<<mean_ptr[1]<<"\n";
    }
    else
    {
        LOG(FATAL)<<"read image mean failed";
    }

    //load net model
    boost::shared_ptr<Net<float> > net(new caffe::Net<float>(proto, phase));
    net->CopyTrainedLayersFrom(model);

    const std::vector<boost::shared_ptr<Layer<float> >> layers = net->layers();
    std::vector<boost::shared_ptr<Blob<float> >> net_blobs = net->blobs();
    std::vector<string> layer_names = net->layer_names();
    std::vector<string> blob_names = net->blob_names();
    boost::shared_ptr<Layer<float> > layer;
    boost::shared_ptr<Blob<float> > blob;
    //show input blob size
    Blob<float>* input_blobs = net->input_blobs()[0];
    std::cout<<"\nInput blob size:\n";
    std::cout<<input_blobs->num()<<"\t"<<input_blobs->height()<<"\t"<<input_blobs->width()<<"\n";

    //处理每一层的blob
    const float *mem_ptr;
    CHECK(layers.size()==layer_names.size());
    std::cout<<"\n#Layers: "<<layers.size()<<std::endl;
    std::vector<boost::shared_ptr<Blob<float> >> layer_blobs;
    for(int i=0;i<layers.size();++i)
    {
        layer_blobs = layers[i]->blobs();
        std::cout<<"\n["<<i+1<<"] layer name:"<<layer_names[i]<<", type:"<<layers[i]->type()<<std::endl;
        std::cout<<"#Blobs: "<<layer_blobs.size()<<std::endl;
        for(int j=0;j<layer_blobs.size();++j)
        {
            blob = layer_blobs[j];
            std::cout<<blob->num()<<"\t"<<blob->channels()<<"\t"<<blob->height()<<"\t"<<blob->width()<<"\n";
            mem_ptr = (const float *)blob->cpu_data();
            std::cout<<mean_ptr[0]<<"\t"<<mean_ptr[1]<<"\n";
        }
    }

    //get weights from layer name
    char *query_layer_name = "conv1";
    const float *weight_ptr, *bias_ptr;
    unsigned int layer_id = get_layer_index(net, query_layer_name);
    layer = net->layers()[layer_id];
    std::vector<boost::shared_ptr<Blob<float> >>blobs = layer->blobs();
    if(blobs.size()>0)
    {
        std::cout<<"\nWeights and bias from layer: "<<query_layer_name<<"\n";
        weight_ptr = (const float *)blobs[0]->cpu_data();
        std::cout<<weight_ptr[0]<<"\t"<<weight_ptr[1]<<"\n";
        bias_ptr = (const float *)blobs[1]->cpu_data();
        std::cout<<bias_ptr[0]<<"\t"<<bias_ptr[1]<<"\n";
    }

    //get feature from name
    char *query_blob_name = "conv1";
    unsigned int blob_id = get_blob_index(net, query_blob_name);
    blob = net->blobs()[blob_id];
    unsigned int num_data = blob->count();
    mem_ptr = (const float *)blob->cpu_data();
    std::cout<<"\n#Features: "<<num_data<<"\n";
    std::cout<<mean_ptr[0]<<"\t"<<mean_ptr[1]<<"\n";

    std::cout<<"End"<<std::endl;
    return 0;
}
