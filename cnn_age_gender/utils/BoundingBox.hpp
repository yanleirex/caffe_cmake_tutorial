//
// Created by yanlei on 16-10-19.
//

#ifndef CAFFE_CMAKE_BOUNDINGBOX_HPP
#define CAFFE_CMAKE_BOUNDINGBOX_HPP

#include <vector>
#include <opencv2/opencv.hpp>


class BoundingBox {
    friend void nms_average(std::vector<BoundingBox>&, std::vector<BoundingBox>&, float);

    friend void nms_max(std::vector<BoundingBox>&, std::vector<BoundingBox>&, float);

    friend bool sort_by_confidence_reverse(const BoundingBox&, const BoundingBox&);

    friend bool sort_by_size(const BoundingBox&, const BoundingBox&);

public:
    BoundingBox(float x, float y, float width, float height, float prob) : x(x), y(y), width(width), height(height), prob(prob)
    {

    }
    BoundingBox(BoundingBox&&) = default;
    BoundingBox(const BoundingBox&) = default;

    BoundingBox& operator=(const BoundingBox&) = default;
    BoundingBox& operator=(BoundingBox&&)= default;
    BoundingBox() = default;

    ~BoundingBox() = default;

    /*
     * change the bounding box to cv::Rect ignoring the prob
     */
    cv::Rect transformToCVRect()
    {
        return cv::Rect(x, y, width, height);
    }

    float getX() const
    {
        return x;
    }

    float getY() const
    {
        return y;
    }

    float getWidth() const
    {
        return width;
    }

    float getHeight() const
    {
        return height;
    }

    float getProb() const
    {
        return prob;
    }

private:
    float area() const {
        return width*height;
    }

    float x;
    float y;
    float width;
    float height;
    float prob;
};

bool sort_by_confidence_reverse(const BoundingBox& a, const BoundingBox& b);

bool sort_by_size(const BoundingBox&, const BoundingBox&);

void nms_average(std::vector<BoundingBox>&, std::vector<BoundingBox>&, float);

void nms_max(std::vector<BoundingBox>&, std::vector<BoundingBox>&, float);


#endif //CAFFE_CMAKE_BOUNDINGBOX_H
