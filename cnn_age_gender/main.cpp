//
// Created by yanlei on 16-10-19.
//
#include <iostream>

#include "face_detection/face_detection.hpp"
#include "age_estimation/age_estimation.hpp"
#include "gender_estimation/gender_estimation.hpp"

int main(int argc, const char* argv[])
{
    cv::Mat image = cv::imread(project_root+"tmp.jpg");
    std::vector<BoundingBox> face_area;
    face_detecton(image, face_area);

    std::vector<Gender> genders(gender_estimation(image, face_area));
    std::vector<Age> ages(age_estimation(image, face_area));

    auto i=0;
    for(const auto& roi:face_area)
    {
        cv::Rect head(extend_face_to_whole_head(roi, image.rows, image.cols).transformToCVRect());

        cv::rectangle(image, head, cv::Scalar(255, 255, 255));
        cv::putText(image, gender_lists[genders[i]], cv::Point(head.x, head.y+10), 0, 0.7, cv::Scalar(0, 0, 255), 2);
        cv::putText(image, age_list[ages[i++]], cv::Point(head.x, head.y+head.height), 0, 0.7, cv::Scalar(255, 0, 0), 2);
    }

    cv::imshow("Ds", image);
    cv::waitKey();
    return 0;
}

