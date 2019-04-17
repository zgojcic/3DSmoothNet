#ifndef _TOLDI_H_
#define _TOLDI_H_
#define Pi 3.1415926
#define NULL_POINTID -1
#define TOLDI_NULL_PIXEL 100
//
typedef pcl::PointCloud<pcl::PointXYZ>::Ptr PointCloudPtr;
typedef pcl::PointXYZ PointInT;
typedef struct
{
    float x;
    float y;
    float z;
} Vertex;
typedef struct
{
    int pointID;
    Vertex x_axis;
    Vertex y_axis;
    Vertex z_axis;
} LRF;

#include <pcl/surface/gp3.h>
#include <pcl/surface/mls.h>

//
typedef pcl::PointCloud<pcl::PointXYZ>::Ptr PointCloudPtr;
typedef pcl::PointXYZ PointT;
typedef pcl::PointCloud<pcl::PointXYZ> PointCloud;
typedef pcl::PointNormal PointNormalT;
typedef pcl::PointCloud<PointNormalT> PointCloudWithNormals;

//	Utility functions
bool processCommandLine(int argc, char** argv, std::string &file_cloud, float &support_radius, int &num_voxels, float &smoothing_kernel_width, std::string &file_keypoints, std::string &output_folder);
std::vector<int> readKeypoints(std::string filename);
bool fileExist(const std::string& name);
void saveVector(std::string filename, const std::vector<std::vector<float>> descriptor);
flann::Matrix<float> initializeGridMatrix(const int n, float x_step, float y_step, float z_step);

//	TOLDI_LRF functions
void toldiComputeZaxis(pcl::PointCloud<pcl::PointXYZ>::Ptr cloud, Vertex &z_axis, std::vector<float>pointDst);
void toldiComputeXaxis(pcl::PointCloud<pcl::PointXYZ>::Ptr cloud,Vertex z_axis,float sup_radius, std::vector<float> PointDist,Vertex &x_axis);
void toldiComputeYaxis(Vertex x_axis,Vertex z_axis,Vertex &y_axis);
void toldiComputeLRF(pcl::PointCloud<pcl::PointXYZ>::Ptr cloud, std::vector<int> indices, float sup_radius, float smoothingFactor, std::vector<LRF>&Cloud_LRF, std::vector<std::vector <int>> &Neighbors, std::vector<std::vector <int>> &NeighborsSmoothingIDX, std::vector<std::vector <float>> &NeighborsSmoothingDistance);

//	SDV computation
void transformCloud(pcl::PointCloud<pcl::PointXYZ>::Ptr cloud,LRF pointLRF,pcl::PointCloud<pcl::PointXYZ>::Ptr &transformed_cloud);
void computeLocalDepthFeature(pcl::PointCloud<pcl::PointXYZ>::Ptr cloud, std::vector<int> indices, std::vector<std::vector<int>> indices_neighbors, std::vector<LRF> cloud_LRF, float sup_radius, flann::Matrix<float> voxel_coordinates, int num_voxels, float smoothingFactor, std::string saveFileName);


#endif