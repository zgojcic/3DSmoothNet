/**
3DSmoothNet
core.cpp

Purpose: containes all the functions used to generate the  Smoothed Density Value voxel representation
of the interest points neighborhood

@Author : Zan Gojcic, Caifa Zhou
@Version : 1.0

*/

#include <boost/filesystem.hpp>
#include <boost/program_options.hpp>
#include <pcl/common/centroid.h>
#include <pcl/filters/extract_indices.h>
#include <omp.h>
#include "core.h"


namespace po = boost::program_options;


/*
	Utility functions
*/

// Processes command line arguments and returns the default values if the parameters are not specified by the user
bool processCommandLine(int argc, char** argv,
	std::string &file_cloud,
	float &support_radius,
	int &num_voxels,
	float &smoothing_kernel_width,
	std::string &file_keypoints,
	std::string &output_folder)
{
	try
	{
		po::options_description desc("Allowed options");
		desc.add_options()
			("help", "Given a point cloud and a support radius, this code generates a SDV voxel representation for the selected interest points.")
			("fileCloud,f", po::value<std::string>(&file_cloud)->required(), "Input point cloud file in .ply format")
			("supportRadius,r", po::value<float>(&support_radius)->default_value(0.150),
				"Half size of the voxel grid.")
			("numVoxels,n", po::value<int>(&num_voxels)->default_value(16),
				"Number of voxels in a side of the grid. Whole grid is nxnxn.")
			("smoothingKernelWidth,h", po::value<float>(&smoothing_kernel_width)->default_value(1.75),
				"Width of the Gaussia kernel used for smoothing.")
			("fileKeypoints,k", po::value<std::string>(&file_keypoints)->default_value("0"),
				"Path to the file with the indices of the interest points. If 0, SDV voxel grid representation if computed for all the points")
			("outputFolder,o", po::value<std::string>(&output_folder)->default_value("./data/sdv_voxel_grid/"),
				"Output folder path.")
			;

		po::variables_map vm;
		po::store(po::parse_command_line(argc, argv, desc), vm);

		if (vm.count("help"))
		{
			std::cout << desc << "\n";
			return false;
		}

		po::notify(vm);
	}
	catch (std::exception& e)
	{
		std::cerr << "ERROR: " << e.what() << "\n";
		return false;
	}
	catch (...)
	{
		std::cerr << "Unknown error!" << "\n";
		return false;
	}

	return true;
}

// Reads a file with the interest point indices file
std::vector<int> readKeypoints(std::string filename)
{
    char separator = ' ';
    std::vector<int> result;
    std::string row, item;

    std::ifstream in(filename);

    while (getline(in, row))
    {
        std::stringstream ss(row);
        std::getline(ss, item, separator);
        result.push_back(std::stoi(item.c_str()));
    }

    in.close();
    return result;
}

// Checks if the file exists
bool fileExist(const std::string& name)
{
    std::ifstream f(name.c_str());  // New enough C++ library will accept just name
    return f.is_open();
}

// Saves the descriptor to a binary csv file
void saveVector(std::string filename, const std::vector<std::vector<float>> descriptor)
{
    std::cout << "Saving Features to a CSV file:" << std::endl;
    std::cout << filename << std::endl;

    std::ofstream outFile;
    outFile.open(filename, std::ios::binary);

    float writerTemp;
    for (int i = 0; i < descriptor.size(); i++)
    {
        for (int j = 0; j < descriptor[i].size(); j++)
        {
            writerTemp = descriptor[i][j];
            outFile.write(reinterpret_cast<const char*>(&writerTemp), sizeof(float));
        }
    }
    outFile.close();
}


// Initizales a grid using the step size and the number of voxels per side
flann::Matrix<float> initializeGridMatrix(const int n, float x_step, float y_step, float z_step)
{
    int grid_size = n*n*n;
    flann::Matrix<float> input(new float[grid_size * 3], grid_size, 3);

    float xs = -(n / 2)*x_step + 0.5*x_step;
    float ys = -(n / 2)*y_step + 0.5*y_step;
    float zs = -(n / 2)*z_step + 0.5*z_step;

    for (int i = 0; i < n; i++)
    {
        //move on x axis
        for (int j = 0; j < n; j++)
        {
            //move on y axis
            for (int k = 0; k < n; k++)
            {
                //move on z axis
                input[i + n*j + n*n*k][0] = xs + x_step * i;
                input[i + n*j + n*n*k][1] = ys + y_step * j;
                input[i + n*j + n*n*k][2] = zs + z_step * k;
            }
        }
    }
    return input;
}


/*
	Functions for estimation of the local reference frame
*/

// Estimates the Z axis of the local reference frame
void toldiComputeZaxis(pcl::PointCloud<pcl::PointXYZ>::Ptr cloud, Vertex &z_axis, std::vector<float> point_dst)
{
    int i;
    pcl::PointXYZ query_point = cloud->points[0];
    // calculate covariance matrix
    Eigen::Matrix3f Cov = Eigen::Matrix3f::Zero();
    Eigen::Matrix<float, 4, 1> centroid;
    centroid[0] = query_point.x;
    centroid[1] = query_point.y;
    centroid[2] = query_point.z;

    Eigen::Vector4f queryPointVector = query_point.getVector4fMap();
    Eigen::Matrix<float, Eigen::Dynamic, 4> vij(point_dst.size(), 4);
    int valid_nn_points = 0;
    double distance = 0.0;
    double sum = 0.0;

    pcl::computeCovarianceMatrix(*cloud, centroid, Cov);

    EIGEN_ALIGN16 Eigen::Vector3f::Scalar eigen_min;
    EIGEN_ALIGN16 Eigen::Vector3f normal;
    pcl::eigen33(Cov, eigen_min, normal);
    z_axis.x = normal(0);
    z_axis.y = normal(1);
    z_axis.z = normal(2);

    // z-axis sign disambiguity
    float z_sign = 0;
    for (i = 0; i < cloud->points.size(); i++)
    {
        float vec_x = query_point.x - cloud->points[i].x;
        float vec_y = query_point.y - cloud->points[i].y;
        float vec_z = query_point.z - cloud->points[i].z;
        z_sign += (vec_x*z_axis.x + vec_y*z_axis.y + vec_z*z_axis.z);
    }
    if (z_sign < 0)
    {
        z_axis.x = -z_axis.x;
        z_axis.y = -z_axis.y;
        z_axis.z = -z_axis.z;
    }
}

// Estimates the X axis of the local reference frame
void toldiComputeXaxis(pcl::PointCloud<pcl::PointXYZ>::Ptr cloud, Vertex z_axis, float sup_radius, std::vector<float> point_dst, Vertex &x_axis)
{
    int i, j;
    pcl::PointXYZ query_point = cloud->points[0];
    //
    std::vector<Vertex> vec_proj;
    std::vector<float> dist_weight, sign_weight;//store weights w1,w2
    for (i = 0; i < cloud->points.size(); i++)
    {
        Vertex temp;
        Vertex pq = { cloud->points[i].x - query_point.x,cloud->points[i].y - query_point.y,cloud->points[i].z - query_point.z };
        float proj = z_axis.x*pq.x + z_axis.y*pq.y + z_axis.z*pq.z;
        if (proj >= 0)
            sign_weight.push_back(pow(proj, 2));
        else
            sign_weight.push_back(-pow(proj, 2));
        temp.x = pq.x - proj*z_axis.x;
        temp.y = pq.y - proj*z_axis.y;
        temp.z = pq.z - proj*z_axis.z;
        vec_proj.push_back(temp);
    }

    for (i = 0; i < point_dst.size(); i++)
    {
        float wei_temp = sup_radius - point_dst[i];
        wei_temp = pow(wei_temp, 2);
        dist_weight.push_back(wei_temp);
    }
    Vertex x_axis_temp = { 0.0f,0.0f,0.0f };
    for (i = 0; i < cloud->points.size(); i++)
    {
        float weight_sum = dist_weight[i] * sign_weight[i];
        x_axis_temp.x += weight_sum*vec_proj[i].x;
        x_axis_temp.y += weight_sum*vec_proj[i].y;
        x_axis_temp.z += weight_sum*vec_proj[i].z;
    }
    //Normalization
    float size = sqrt(pow(x_axis_temp.x, 2) + pow(x_axis_temp.y, 2) + pow(x_axis_temp.z, 2));
    x_axis_temp.x /= size;
    x_axis_temp.y /= size;
    x_axis_temp.z /= size;
    x_axis = x_axis_temp;
}

// Estimates the Y axis of the local reference frame
void toldiComputeYaxis(Vertex x_axis, Vertex z_axis, Vertex &y_axis)
{
    Eigen::Vector3f x(x_axis.x, x_axis.y, x_axis.z);
    Eigen::Vector3f z(z_axis.x, z_axis.y, z_axis.z);
    Eigen::Vector3f y;

    y = x.cross(z);//cross product

    y_axis.x = y(0);
    y_axis.y = y(1);
    y_axis.z = y(2);
}

// Compute the lrf accordin to the method from toldi paper, for the points selected with the indices
void toldiComputeLRF(pcl::PointCloud<pcl::PointXYZ>::Ptr cloud,
                     std::vector<int> indices,
                     float sup_radius,
                     float smoothingFactor,
                     std::vector<LRF> &cloud_LRF,
                     std::vector<std::vector <int>>& neighbors,
                     std::vector<std::vector <int>>& neighbors_smoothing_idx,
                     std::vector<std::vector <float>>& neighbors_smoothing_distance)
{
    int i, j, m;
    // Initialize all the variables
    pcl::KdTreeFLANN<pcl::PointXYZ> kdtree;
    std::vector<int> point_idx;
    std::vector<float> point_dst;
    kdtree.setInputCloud(cloud);
    pcl::PointXYZ query_point;
    pcl::PointXYZ test;
    //LRF calculation
    for (i = 0; i < indices.size(); i++)
    {
        pcl::PointCloud<pcl::PointXYZ>::Ptr sphere_neighbor(new pcl::PointCloud<pcl::PointXYZ>);//local surface
        pcl::PointCloud<pcl::PointXYZ>::Ptr sphere_neighbor_z(new pcl::PointCloud<pcl::PointXYZ>);//local surface for computing the z-axis of LRF
        query_point = cloud->points[indices[i]];

        if (kdtree.radiusSearch(query_point, sup_radius, point_idx, point_dst) > 10)//only if there are more than 10 points in the local surface
        {

            for (j = 0; j < point_idx.size(); j++)
            {

                test.x = cloud->points[point_idx[j]].x - query_point.x;
                test.y = cloud->points[point_idx[j]].y - query_point.y;
                test.z = cloud->points[point_idx[j]].z - query_point.z;

                sphere_neighbor_z->points.push_back(test);
            }

            // Save points for feature computation
			neighbors.at(indices[i]) = point_idx;

            std::transform(point_dst.begin(), point_dst.end(), point_dst.begin(), std::ptr_fun<float, float>(std::sqrt));
            // Find first element that has a distance bigger than the smoothing threshold
            auto lower = std::lower_bound(point_dst.begin(), point_dst.end(), smoothingFactor);
            // Index of the last element smaller then the threshold
            int  index_last_element = lower - point_dst.begin();

            // Copy neighbours to vector
            std::vector<int> point_idx_smoothing(&point_idx[0], &point_idx[index_last_element]);
            std::vector<float> point_distance_smoothing(&point_dst[0], &point_dst[index_last_element]);

			neighbors_smoothing_idx.at(indices[i]) = point_idx_smoothing;
			neighbors_smoothing_distance.at(indices[i]) = point_distance_smoothing;


            for (j = 0; j < point_idx.size(); j++)
            {
                sphere_neighbor->points.push_back(cloud->points[point_idx[j]]);
            }

            Vertex x_axis, y_axis, z_axis;
            toldiComputeZaxis(sphere_neighbor_z, z_axis, point_dst);
            toldiComputeXaxis(sphere_neighbor, z_axis, sup_radius, point_dst, x_axis);
            toldiComputeYaxis(x_axis, z_axis, y_axis);
            LRF temp = { indices[i],x_axis,y_axis,z_axis };
			cloud_LRF.at(indices[i]) = temp;
        }
        else
        {
            std::cout << "Less then ten points in the neighborhood!!!" << std::endl;
            LRF temp = { NULL_POINTID,{0.0f,0.0f,0.0f},{0.0f,0.0f,0.0f},{0.0f,0.0f,0.0f} };
			cloud_LRF.at(indices[i]) = temp;
        }
    }
}




/*
	Functions used for computation of the SDV voxel grid representation
*/

// transform the local neighbhorhood of the selecred interest point to its canonical representation defined by the estimated local reference frame
void transformCloud(pcl::PointCloud<pcl::PointXYZ>::Ptr cloud, LRF pointLRF, pcl::PointCloud<pcl::PointXYZ>::Ptr &transformed_cloud)
{
    pcl::PointXYZ point = cloud->points[0]; //the centroid of the local surface
    int number_of_points = cloud->points.size() - 1; // remove the first point which is centroid
    transformed_cloud->points.resize(number_of_points);
    Eigen::Matrix3f matrix;
    matrix(0, 0) = pointLRF.x_axis.x;
    matrix(0, 1) = pointLRF.x_axis.y;
    matrix(0, 2) = pointLRF.x_axis.z;
    matrix(1, 0) = pointLRF.y_axis.x;
    matrix(1, 1) = pointLRF.y_axis.y;
    matrix(1, 2) = pointLRF.y_axis.z;
    matrix(2, 0) = pointLRF.z_axis.x;
    matrix(2, 1) = pointLRF.z_axis.y;
    matrix(2, 2) = pointLRF.z_axis.z;

    // Iterrate over all the points and save the transfomed version (+1 because one point is ommited at the start )
    for (int i = 0; i < number_of_points; i++)
    {
        Eigen::Vector3f transformed_point(
            cloud->points[i + 1].x - point.x,
            cloud->points[i + 1].y - point.y,
            cloud->points[i + 1].z - point.z);

        transformed_point = matrix * transformed_point;

        pcl::PointXYZ new_point;
        new_point.x = transformed_point(0);
        new_point.y = transformed_point(1);
        new_point.z = transformed_point(2);
        transformed_cloud->points[i] = new_point;
    }
}



// estimate the SDV voxel grid for all interes points
void computeLocalDepthFeature(pcl::PointCloud<pcl::PointXYZ>::Ptr cloud,
                              std::vector<int> evaluation_points,
                              std::vector<std::vector<int>> indices_neighbors,
                              std::vector<LRF> cloud_LRF,
                              float sup_radius,
                              flann::Matrix<float> voxel_coordinates,
                              int num_voxels,
                              float smoothing_factor,
                              std::string saveFileName)
{
    // Iterrate over all the points for which the descriptor is to be computed
    pcl::PointXYZ queryPoint;
    pcl::ExtractIndices<pcl::PointXYZ> extract;
    int counter_voxel = num_voxels * num_voxels * num_voxels;

    // Create the filtering object
    std::string saving_path_file;
    int threads_ = omp_get_max_threads();
    std::cout << "Starting SDV computation!" << std::endl;
    std::cout << threads_ << " threads will be used!!" << std::endl;

    // Initialize the space for the SDV values
    std::vector <std::vector <float>> DIMATCH_Descriptor(evaluation_points.size(), std::vector<float>(counter_voxel, 0));

    // Initialize the point to the descriptor for each thread used
    Eigen::VectorXf *descriptor = new Eigen::VectorXf[threads_];


    for (int i = 0; i < threads_; i++)
    {
        descriptor[i].setZero(counter_voxel);
    }

    int tid;

    // Create path for saving
    const char* path = "data";
    boost::filesystem::path dir(path);
    boost::filesystem::create_directory(dir);
    int progress_counter = 0;

    #pragma omp parallel for shared(cloud,evaluation_points,indices_neighbors,counter_voxel,DIMATCH_Descriptor,progress_counter) private(tid,extract)  num_threads(threads_)
    for (int i = 0; i < evaluation_points.size(); i++)
    {

        pcl::PointCloud<pcl::PointXYZ>::Ptr sphere_neighbors(new pcl::PointCloud<pcl::PointXYZ>);
        pcl::PointCloud<pcl::PointXYZ>::Ptr sphere_neighbors_transformed(new pcl::PointCloud<pcl::PointXYZ>);
        pcl::PointIndices::Ptr inliers(new pcl::PointIndices());
        pcl::KdTreeFLANN<pcl::PointXYZ> kdtree;
        std::vector<std::vector<size_t>> indices;
        std::vector<std::vector<float>> dists;

        tid = omp_get_thread_num();
        descriptor[tid].setZero(counter_voxel);
        // Save query points coordinates
        queryPoint = cloud->points[evaluation_points[i]];

        // Extract neighbors from point cloud
        extract.setInputCloud(cloud);
        inliers->indices = indices_neighbors[evaluation_points[i]];
        extract.setIndices(inliers);
        extract.setNegative(false);
        extract.filter(*sphere_neighbors);


        // Transform the neighbors to the local reference frame
        if (sphere_neighbors->points.size() != 0)
        {
            transformCloud(sphere_neighbors, cloud_LRF[evaluation_points[i]], sphere_neighbors_transformed);

            flann::Matrix<float> data(new float[sphere_neighbors_transformed->size() * 3], sphere_neighbors_transformed->size(), 3);



            for (int j = 0; j < sphere_neighbors_transformed->size(); j++)
            {
                data[j][0] = sphere_neighbors_transformed->points[j].x;
                data[j][1] = sphere_neighbors_transformed->points[j].y;
                data[j][2] = sphere_neighbors_transformed->points[j].z;
            }

            flann::SearchParams search_parameters = flann::SearchParams();
            search_parameters.checks = -1;
            search_parameters.sorted = false;
            search_parameters.use_heap = flann::FLANN_True;


            flann::KDTreeSingleIndexParams index_parameters = flann::KDTreeSingleIndexParams();
            flann::KDTreeSingleIndex<flann::L2_3D<float> > index(data, index_parameters);
            index.buildIndex();

            // Square the smoothing factor as flann uses sqaured distances
            float smoothing_factor_sqrd = 9 * smoothing_factor*smoothing_factor;
            index.radiusSearch(voxel_coordinates, indices, dists, smoothing_factor_sqrd, search_parameters);

            // Computes the normalization term and the variance
            double normalization_term = 1 / (sqrt(2 * M_PI) * smoothing_factor);
            double variance_term = -1 / (2 * (smoothing_factor * smoothing_factor));


            for (int voxel_idx = 0; voxel_idx < counter_voxel - 1; voxel_idx++)
            {
                //cout << " thread: " << omp_get_thread_num() << endl;
                // Extract all the distances
                std::vector<float> point_distances = dists[voxel_idx];
                if (!point_distances.empty())
                {
                    // Multiply with variance term
                    std::transform(point_distances.begin(), point_distances.end(), point_distances.begin(),
                                   std::bind1st(std::multiplies<float>(), variance_term));

                    // Exponent
                    std::transform(point_distances.begin(), point_distances.end(), point_distances.begin(), std::ptr_fun<float, float>(std::exp));

                    // Normalization term
                    std::transform(point_distances.begin(), point_distances.end(), point_distances.begin(),
                                   std::bind1st(std::multiplies<float>(), normalization_term));

                    // Sum the elements
                    float sum_of_elems = std::accumulate(point_distances.begin(), point_distances.end(), 0.0f) / point_distances.size();

                    // Sum up all the depths value to the voxel depth value
                    descriptor[tid][voxel_idx] = sum_of_elems;

                }
                else
                {
                    // if no points in the neighborhood set the voxel depth to 0
                    descriptor[tid][voxel_idx] = 0;
                }

            }

            // Normalize the descriptor
            float descriptor_sum = descriptor[tid].sum();
            if (descriptor_sum != 0)
                descriptor[tid] = descriptor[tid] / descriptor_sum;

            for (int d = 0; d < descriptor[tid].size(); ++d)
                DIMATCH_Descriptor[i][d] = descriptor[tid][d];

        }
        else
        {
            for (int voxel_idx = 0; voxel_idx < counter_voxel - 1; voxel_idx++)
            {
                descriptor[tid][voxel_idx] = 0;
            }

            for (int d = 0; d < descriptor[tid].size(); ++d)
                DIMATCH_Descriptor[i][d] = descriptor[tid][d];

        }
    }

    // Create the name with the radius, number of voxels and the smoothing factor
    saving_path_file = saveFileName + "_" + std::to_string(sup_radius) + "_" + std::to_string(num_voxels) + "_" + std::to_string(smoothing_factor * num_voxels / sup_radius) + ".csv";


    // Write descriptor to CSV file
    saveVector(saving_path_file, DIMATCH_Descriptor);
}
