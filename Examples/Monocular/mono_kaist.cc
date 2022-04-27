#include <iostream>
#include <algorithm>
#include <fstream>
#include <chrono>
#include <iomanip>

#include <opencv2/core/core.hpp>

#include "System.h"

using namespace std;

int main(int argc, char **argv)
{
    if (argc != 4)
    {
        cerr << endl
             << "Usage: ./mono_kaist path_to_vocabulary path_to_settings path_to_kaist" << endl;
        return 1;
    }

    // Retrieve paths to images
    std::string sData_path;
    sData_path = argv[3];                                                //数据集路径
    string data_stamp_path = sData_path + "/sensor_data/data_stamp.csv"; //数据<时间戳,数据类型>
    ifstream file_data_stamp(data_stamp_path);
    if (!file_data_stamp.is_open())
    {
        cerr << "[PublishData]: Failed to open data_stamp file.";
        return -1;
    }
    std::cout << " times ";

    vector<double> vTimestamps; //图像的时间戳

    int nImages = 0;

    // Create SLAM system. It initializes all system threads and gets ready to process frames.
    ORB_SLAM3::System SLAM(argv[1], argv[2], ORB_SLAM3::System::MONOCULAR, true);
    float imageScale = SLAM.GetImageScale();

    // Vector for tracking time statistics
    vector<float> vTimesTrack;

    cout << endl
         << "-------" << endl;
    cout << "Start processing sequence ..." << endl;
    cout << "Images in the sequence: " << nImages << endl
         << endl;

    // Main loop
    double t_resize = 0.0f;
    double t_track = 0.0f;
    cv::Mat im;
    vector<string> line_data_vec;
    line_data_vec.reserve(17);
    string line_str, value_str;
    int ni = 0;
    while (getline(file_data_stamp, line_str))
    {
        line_data_vec.clear();
        stringstream ss(line_str); //取一个时间戳
        while (getline(ss, value_str, ','))
        {
            line_data_vec.push_back(value_str); //存储该时间戳对应的，时间戳数据、传感器类型
        }
        constexpr double kToSecond = 1e-9;
        string time_str = line_data_vec[0];
        double timestamp = stod(time_str) * kToSecond; //计算当前时间戳转化到秒单位
        string img_file;

        const string &sensor_type = line_data_vec[1]; //传感器类型

        if (sensor_type == "stereo") //发布图像topic
        {
            img_file = sData_path + "/image/stereo_left/" + time_str + ".png";
            vTimestamps.push_back(timestamp);
            nImages++;
        }
        else
        {
            continue;
        }

        // Read image from file
        im = cv::imread(img_file, cv::IMREAD_UNCHANGED); //,cv::IMREAD_UNCHANGED);
        double tframe = timestamp;

        if (im.empty())
        {
            cerr << endl
                 << "Failed to load image at: " << line_data_vec[0] << ":" << sensor_type << endl;
            return 1;
        }

        if (imageScale != 1.f)
        {
#ifdef REGISTER_TIMES
#ifdef COMPILEDWITHC11
            std::chrono::steady_clock::time_point t_Start_Resize = std::chrono::steady_clock::now();
#else
            std::chrono::monotonic_clock::time_point t_Start_Resize = std::chrono::monotonic_clock::now();
#endif
#endif
            int width = im.cols * imageScale;
            int height = im.rows * imageScale;
            cv::resize(im, im, cv::Size(width, height));
#ifdef REGISTER_TIMES
#ifdef COMPILEDWITHC11
            std::chrono::steady_clock::time_point t_End_Resize = std::chrono::steady_clock::now();
#else
            std::chrono::monotonic_clock::time_point t_End_Resize = std::chrono::monotonic_clock::now();
#endif
            t_resize = std::chrono::duration_cast<std::chrono::duration<double, std::milli>>(t_End_Resize - t_Start_Resize).count();
            SLAM.InsertResizeTime(t_resize);
#endif
        }

#ifdef COMPILEDWITHC11
        std::chrono::steady_clock::time_point t1 = std::chrono::steady_clock::now();
#else
        std::chrono::monotonic_clock::time_point t1 = std::chrono::monotonic_clock::now();
#endif

        // Pass the image to the SLAM system
        SLAM.TrackMonocular(im, tframe, vector<ORB_SLAM3::IMU::Point>(), img_file);
        std::cout << img_file << ":" << tframe << std::endl;

#ifdef COMPILEDWITHC11
        std::chrono::steady_clock::time_point t2 = std::chrono::steady_clock::now();
#else
        std::chrono::monotonic_clock::time_point t2 = std::chrono::monotonic_clock::now();
#endif

#ifdef REGISTER_TIMES
        t_track = t_resize + std::chrono::duration_cast<std::chrono::duration<double, std::milli>>(t2 - t1).count();
        SLAM.InsertTrackTime(t_track);
#endif

        double ttrack = std::chrono::duration_cast<std::chrono::duration<double>>(t2 - t1).count();

        vTimesTrack.push_back(ttrack);

        // Wait to load the next frame
        double T = 0;
        if (ni < nImages - 1)
            T = vTimestamps[ni + 1] - tframe;
        else if (ni > 0)
            T = tframe - vTimestamps[ni - 1];

        if (ttrack < T)
            usleep((T - ttrack) * 1e6);

        ni++;
    }

    // Stop all threads
    SLAM.Shutdown();

    // Tracking time statistics
    sort(vTimesTrack.begin(), vTimesTrack.end());
    float totaltime = 0;
    for (int ni = 0; ni < nImages; ni++)
    {
        totaltime += vTimesTrack[ni];
    }
    cout << "-------" << endl
         << endl;
    cout << "median tracking time: " << vTimesTrack[nImages / 2] << endl;
    cout << "mean tracking time: " << totaltime / nImages << endl;

    // Save camera trajectory
    SLAM.SaveKeyFrameTrajectoryTUM("KeyFrameTrajectory.txt");

    return 0;
}
