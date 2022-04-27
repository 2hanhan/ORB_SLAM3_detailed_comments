#include <iostream>
#include <algorithm>
#include <fstream>
#include <chrono>
#include <iomanip>

#include <opencv2/core/core.hpp>

#include "System.h"

using namespace std;

bool LoadSensorData(string &sensor_data_file, unordered_map<string, string> *time_data_map);
void LoadKaist(const string &Data_path,
               vector<string> &vstrImages, vector<double> &vTimestampsCam,
               vector<double> &vTimestampsImu, vector<cv::Point3f> &vAcc, vector<cv::Point3f> &vGyro);

int main(int argc, char **argv)
{
    if (argc != 4)
    {
        cerr << endl
             << "Usage: ./mono_kaist path_to_vocabulary path_to_settings path_to_kaist " << endl;
        return 1;
    }

    std::string Data_path;
    Data_path = argv[3];

    const int num_seq = 1;
    // Load all sequences:
    int seq;
    vector<vector<string>> vstrImageFilenames;
    vector<vector<double>> vTimestampsCam;
    vector<vector<cv::Point3f>> vAcc, vGyro;
    vector<vector<double>> vTimestampsImu;
    vector<int> nImages;
    vector<int> nImu;
    vector<int> first_imu(num_seq, 0);

    vstrImageFilenames.resize(num_seq);
    vTimestampsCam.resize(num_seq);
    vAcc.resize(num_seq);
    vGyro.resize(num_seq);
    vTimestampsImu.resize(num_seq);
    nImages.resize(num_seq);
    nImu.resize(num_seq);

    int tot_images = 0;
    for (seq = 0; seq < num_seq; seq++)
    {

        LoadKaist(Data_path,
                  vstrImageFilenames[seq], vTimestampsCam[seq],
                  vTimestampsImu[seq], vAcc[seq], vGyro[seq]);

        cout << "LOADED KAIST!" << endl;

        nImages[seq] = vstrImageFilenames[seq].size();
        tot_images += nImages[seq];
        nImu[seq] = vTimestampsImu[seq].size();

        if ((nImages[seq] <= 0) || (nImu[seq] <= 0))
        {
            cerr << "ERROR: Failed to load images or IMU for sequence" << seq << endl;
            return 1;
        }

        // Find first imu to be considered, supposing imu measurements start first

        while (vTimestampsImu[seq][first_imu[seq]] <= vTimestampsCam[seq][0])
            first_imu[seq]++;
        first_imu[seq]--; // first imu measurement to be considered
    }

    // Vector for tracking time statistics
    vector<float> vTimesTrack;
    vTimesTrack.resize(tot_images);

    cout.precision(17);

    // Create SLAM system. It initializes all system threads and gets ready to process frames.
    ORB_SLAM3::System SLAM(argv[1], argv[2], ORB_SLAM3::System::IMU_MONOCULAR, true);
    float imageScale = SLAM.GetImageScale();

    double t_resize = 0.f;
    double t_track = 0.f;

    int proccIm = 0;
    for (seq = 0; seq < num_seq; seq++)
    {

        // Main loop
        cv::Mat im;
        vector<ORB_SLAM3::IMU::Point> vImuMeas;
        proccIm = 0;
        for (int ni = 0; ni < nImages[seq]; ni++, proccIm++)
        {
            // Read image from file
            im = cv::imread(vstrImageFilenames[seq][ni], cv::IMREAD_UNCHANGED); // CV_LOAD_IMAGE_UNCHANGED);

            double tframe = vTimestampsCam[seq][ni];

            if (im.empty())
            {
                cerr << endl
                     << "Failed to load image at: "
                     << vstrImageFilenames[seq][ni] << endl;
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

            // Load imu measurements from previous frame
            vImuMeas.clear();

            if (ni > 0)
            {
                // cout << "t_cam " << tframe << endl;

                while (vTimestampsImu[seq][first_imu[seq]] <= vTimestampsCam[seq][ni])
                {
                    vImuMeas.push_back(ORB_SLAM3::IMU::Point(vAcc[seq][first_imu[seq]].x, vAcc[seq][first_imu[seq]].y, vAcc[seq][first_imu[seq]].z,
                                                             vGyro[seq][first_imu[seq]].x, vGyro[seq][first_imu[seq]].y, vGyro[seq][first_imu[seq]].z,
                                                             vTimestampsImu[seq][first_imu[seq]]));
                    first_imu[seq]++;
                }
            }

#ifdef COMPILEDWITHC11
            std::chrono::steady_clock::time_point t1 = std::chrono::steady_clock::now();
#else
            std::chrono::monotonic_clock::time_point t1 = std::chrono::monotonic_clock::now();
#endif

            // Pass the image to the SLAM system
            cout << "tframe = " << tframe << endl;
            SLAM.TrackMonocular(im, tframe, vImuMeas); // TODO change to monocular_inertial

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
            // std::cout << "ttrack: " << ttrack << std::endl;

            vTimesTrack[ni] = ttrack;

            // Wait to load the next frame
            double T = 0;
            if (ni < nImages[seq] - 1)
                T = vTimestampsCam[seq][ni + 1] - tframe;
            else if (ni > 0)
                T = tframe - vTimestampsCam[seq][ni - 1];

            if (ttrack < T)
                usleep((T - ttrack) * 1e6); // 1e6
        }
        if (seq < num_seq - 1)
        {
            cout << "Changing the dataset" << endl;

            SLAM.ChangeDataset();
        }
    }

    // Stop all threads
    SLAM.Shutdown();

    // Save camera trajectory
    SLAM.SaveKeyFrameTrajectoryTUM("Save/CameraTrajectory.txt");
    // Save camera trajectory
    SLAM.SaveKeyFrameTrajectoryTUM("KeyFrameTrajectory.txt");

    return 0;
}

bool LoadSensorData(string &sensor_data_file, unordered_map<string, string> *time_data_map)
{
    ifstream data_file(sensor_data_file);
    if (!data_file.is_open())
    {
        cerr << "[LoadSensorData]: Failed to open sensor data file.";
        return false;
    }
    string line_str, time_str;
    while (getline(data_file, line_str))
    {
        stringstream ss(line_str);
        if (!getline(ss, time_str, ','))
        {
            cerr << "[LoadSensorData]: Find a bad line in the file.: " << line_str;
            return false;
        }
        time_data_map->emplace(time_str, line_str);
    }
    return true;
}

void LoadKaist(const string &Data_path,
               vector<string> &vstrImages, vector<double> &vTimestampsCam,
               vector<double> &vTimestampsImu, vector<cv::Point3f> &vAcc, vector<cv::Point3f> &vGyro)
{
    // IMU信息
    unordered_map<string, string> time_imu_map; // imu的<时间戳,数据>
    string imu_data_path = Data_path + "/sensor_data/xsens_imu.csv";
    if (!LoadSensorData(imu_data_path, &time_imu_map))
    {
        cerr << "Failed to load imu data.";
    }

    //数据集时间同步信息
    string data_stamp_path = Data_path + "/sensor_data/data_stamp.csv"; //数据<时间戳,数据类型>
    ifstream file_data_stamp(data_stamp_path);
    if (!file_data_stamp.is_open())
    {
        cerr << "Failed to open data_stamp file.";
    }
    vector<string> line_data_vec;
    line_data_vec.reserve(17);
    string line_str, value_str;

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

        const string &sensor_type = line_data_vec[1]; //传感器类型
        if (sensor_type == "stereo")                  //图像
        {
            vstrImages.push_back(Data_path + "/image/stereo_left/" + time_str + ".png");
            vTimestampsCam.push_back(timestamp);
        }
        else if (sensor_type == "imu") // IMU
        {
            if (time_imu_map.find(time_str) == time_imu_map.end())
            {
                cerr << "[PublishData]: Failed to find imu data at time: " << time_str;
            }
            const string &imu_str = time_imu_map.at(time_str); //根据时间戳索引imu数据
            stringstream imu_ss(imu_str);
            line_data_vec.clear();
            while (getline(imu_ss, value_str, ','))
            {
                line_data_vec.push_back(value_str);
            }

            vTimestampsImu.push_back(timestamp);
            vAcc.push_back(cv::Point3f(std::stod(line_data_vec[11]), std::stod(line_data_vec[12]), std::stod(line_data_vec[13])));
            vGyro.push_back(cv::Point3f(std::stod(line_data_vec[8]), std::stod(line_data_vec[9]), std::stod(line_data_vec[10])));
        }
        else
        {
            continue;
        }
    }
}
