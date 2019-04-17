/**
3DSmoothNet
main.cpp

Purpose: executes the computation of the SDV voxel grid for the selected interes points

@Author : Zan Gojcic, Caifa Zhou
@Version : 1.0
*/

#include <chrono>
#include <pcl/io/ply_io.h>
#include "core/core.h"


int main(int argc, char *argv[])
{
    // Turn off the warnings of pcl
    pcl::console::setVerbosityLevel(pcl::console::L_ALWAYS);

    // Initialize the variables
    std::string data_file;
    float radius;
    int num_voxels;
    float smoothing_kernel_width;
    std::string interest_points_file;
    std::string output_folder;

    // Get command line arguments
    bool result = processCommandLine(argc, argv, data_file, radius, num_voxels, smoothing_kernel_width, interest_points_file, output_folder);
    if (!result)
        return 1;

    // Check if the output folder exists, otherwise create it
    boost::filesystem::path dir(output_folder);
    if (boost::filesystem::create_directory(dir))
    {
        std::cerr << "Directory Created: " << output_folder << std::endl;
    }

    // Read in the point cloud using the ply reader
    std::cout << "Config parameters successfully read in!! \n" << std::endl;

    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZ>);
    if (fileExist(data_file))
        pcl::io::loadPLYFile(data_file, *cloud);
    else
    {
        std::cout << "Point cloud file does not exsist or cannot be opened!!" << std::endl;
        return 1;
    }

    std::cout << "File: " << data_file << std::endl;
    std::cout << "Number of Points: " << cloud->size() << std::endl;
    std::cout << "Size of the voxel grid: " << 2 * radius << std::endl; // Multiplied with two as half size is used (corresponding to the radius)
    std::cout << "Number of Voxels: " << num_voxels << std::endl;
    std::cout << "Smoothing Kernel: " << smoothing_kernel_width << std::endl;



    // Specify the parameters of the algorithm
    const int grid_size = num_voxels * num_voxels * num_voxels;
    float voxel_step_size = (2 * radius) / num_voxels;
    float lrf_radius = sqrt(3)*radius; // Such that the circumscribed sphere is obtained

    // Initialize the voxel grid
    flann::Matrix<float> voxel_coordinates = initializeGridMatrix(num_voxels, voxel_step_size, voxel_step_size, voxel_step_size);

    // Compute the local reference frame for all the points
    float smoothing_factor = smoothing_kernel_width * (radius / num_voxels); // Equals half a voxel size so that 3X is 1.5 voxel

    // Check if all the points should be evaluated or only selected  ones
    std::string flag_all_points = "0";
    std::vector<int> evaluation_points;

    // Erase /r at the end of the filename (needed in linux environment)
    interest_points_file.erase(std::remove(interest_points_file.begin(), interest_points_file.end(), '\r'), interest_points_file.end());

    // If the keypoint file is not given initialize the ecaluation points with all the points in the point cloud
    if (!interest_points_file.compare(flag_all_points))
    {
        std::vector<int> ep_temp(cloud->width);
        std::iota(ep_temp.begin(), ep_temp.end(), 0);
        evaluation_points = ep_temp;
        ep_temp.clear();
    }
    else
    {
        if (fileExist(interest_points_file))
        {
            std::vector<int> ep_temp = readKeypoints(interest_points_file);
            evaluation_points = ep_temp;
            ep_temp.clear();
        }
        else
        {
            std::cout << "Keypoint file does not exsist or cannot be opened!!" << std::endl;
            return 1;
        }
    }

    std::cout << "Number of keypoints:" << evaluation_points.size() << "\n" << std::endl;


    // Initialize the variables for the NN search and LRF computation
    std::vector<int> indices(cloud->width);
    std::vector<LRF> cloud_lrf(cloud->width);
    std::iota(indices.begin(), indices.end(), 0);
    std::vector <std::vector <int>> nearest_neighbors(cloud->width);
    std::vector <std::vector <int>> nearest_neighbors_smoothing(cloud->width);
    std::vector <std::vector <float>> nearest_neighbors_smoothing_dist(cloud->width);

    // Compute the local reference frame for the interes points (code adopted from https://www.researchgate.net/publication/310815969_TOLDI_An_effective_and_robust_approach_for_3D_local_shape_description
    // and not optimized)

    auto t1_lrf = std::chrono::high_resolution_clock::now();
    toldiComputeLRF(cloud, evaluation_points, lrf_radius, 3 * smoothing_factor, cloud_lrf, nearest_neighbors, nearest_neighbors_smoothing, nearest_neighbors_smoothing_dist);
    auto t2_lrf = std::chrono::high_resolution_clock::now();

    // Compute the SDV representation for all the points

    std::size_t found = data_file.find_last_of("/");
    std::string temp_token = data_file.substr(found + 1);
    std::size_t found2 = data_file.find_last_of(".");

    std::string save_file_name = temp_token.substr(0, found2);
    save_file_name = output_folder + save_file_name;

    // Start the actuall computation
    auto t1 = std::chrono::high_resolution_clock::now();
    computeLocalDepthFeature(cloud, evaluation_points, nearest_neighbors, cloud_lrf, radius, voxel_coordinates, num_voxels, smoothing_factor, save_file_name);
    auto t2 = std::chrono::high_resolution_clock::now();
    std::cout << "\n---------------------------------------------------------" << std::endl;
    std::cout << "LRF computation took "
              << std::chrono::duration_cast<std::chrono::milliseconds>(t2_lrf - t1_lrf).count()
              << " miliseconds\n";
    std::cout << "SDV computation took "
              << std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1).count()
              << " miliseconds\n";
    std::cout << "---------------------------------------------------------" << std::endl;


    return 0;

}
