# GPU-Octree-Ray-Tracing-for-PCL

PCL Version: 1.8.0

Sample Code:

```
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr point_cloud_ptr (new pcl::PointCloud<pcl::PointXYZRGB>);
    pcl::PointCloud<pcl::PointXYZ>::Ptr point_cloud_ptr_xyz (new pcl::PointCloud<pcl::PointXYZ>);
    pcl::io::loadPCDFile ("xyzrgb.pcd", *point_cloud_ptr);
    pcl::io::loadPCDFile ("xyz.pcd", *point_cloud_ptr_xyz);
    cout << point_cloud_ptr_xyz->points.size() << endl;
    cout << point_cloud_ptr->points.size() << endl;

    vector<int> color_device;
    for(int i = 0; i < point_cloud_ptr->points.size(); i++)
    {
        uint8_t r = point_cloud_ptr->points[i].r;
        uint8_t g = point_cloud_ptr->points[i].g;
        uint8_t b = point_cloud_ptr->points[i].b;
        uint32_t rgb = ((uint32_t)r << 16 | (uint32_t)g << 8 | (uint32_t)b);
        color_device.push_back(rgb);
    }

     //prepare device cloud
    pcl::gpu::Octree::PointCloud cloud_device;
    {
        //start = clock();
        ScopeTime up("upload to gpu");
        cloud_device.upload(point_cloud_ptr_xyz->points);
    }

    cout << "======  Build perfomance =====" << endl;
    // build device octree
    pcl::gpu::Octree octree_device;
    {
        octree_device.setCloud(cloud_device);
        octree_device.setCloudColor(color_device);
        ScopeTime up("gpu-set pointcloud");
    }
    {
        ScopeTime up("gpu-build");
        octree_device.build();
    }
    {
        ScopeTime up("gpu-download");
        octree_device.internalDownload();
    }

    // prepare camera intrinsic matrix
    cv::Mat K = (cv::Mat_<float>(3,3) <<
                   865.556239, 0.0, 270.288017,
                   0.0, 865.556239, 252.256596,
                   0.0, 0.0, 1.0);

    cv::Mat K_inv = K.inv();
    float* K_inv_data = (float*)K_inv.data;

    // raytrace octree
    cv::Mat raytrace_gray_img(480, 640, CV_8U);
    cv::Mat raytrace_depth_img(480, 640, CV_16U);
    octree_device.getRaytracingImage(480, 640, raytrace_gray_img.data, (unsigned short*)raytrace_depth_img.data, K_inv_data, 10);
```
