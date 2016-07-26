/*
* Software License Agreement (BSD License)
*
*  Copyright (c) 2011, Willow Garage, Inc.
*  All rights reserved.
*
*  Redistribution and use in source and binary forms, with or without
*  modification, are permitted provided that the following conditions
*  are met:
*
*   * Redistributions of source code must retain the above copyright
*     notice, this list of conditions and the following disclaimer.
*   * Redistributions in binary form must reproduce the above
*     copyright notice, this list of conditions and the following
*     disclaimer in the documentation and/or other materials provided
*     with the distribution.
*   * Neither the name of Willow Garage, Inc. nor the names of its
*     contributors may be used to endorse or promote products derived
*     from this software without specific prior written permission.
*
*  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
*  "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
*  LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS
*  FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE
*  COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
*  INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
*  BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
*  LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
*  CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
*  LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN
*  ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
*  POSSIBILITY OF SUCH DAMAGE.
*
*  Author: Yi Luo, University of California, San Diego, (yil485@ucsd.edu)
*/

#include "internal.hpp"
#include "pcl/gpu/utils/device/limits.hpp"
#include "pcl/gpu/utils/device/warp.hpp"

#include "utils/morton.hpp"
#include "utils/copygen.hpp"
#include "utils/boxutils.hpp"
#include "utils/scan_block.hpp"
#include "utils/bitonic_sort.hpp"
#include "octree_iterator.hpp"

#include <iostream>
#include <stdio.h>
#include <stdint.h>
#include "time.h"

namespace pcl { namespace device { namespace raytracing
{   
	__global__ void Kernelraytracing(int rows, int cols, unsigned char* color_out, unsigned short* depth_out,
      int* nodes, int* codes,
       int* begs, int* ends,
        float3 minp, float3 maxp,
       const float* points_sorted, const int* indices, const int* color,
        int points_sorted_step, float* K_inv, int resolution)
    {    
        int x = threadIdx.x + blockIdx.x * blockDim.x;
        int y = threadIdx.y + blockIdx.y * blockDim.y;
        int offset = x + y * blockDim.x * gridDim.x;

        // stack definition
        //int MAX_LEVELS_PLUS_ROOT = 11;
        int paths[11];
        int parent[11];
        float t_min[11];
        float t_max[11];
        int idx[11];
        int level;

        // init stack
        level = 0;
        paths[level] = (0 << 8) + 1;

        // global variable for raycasting
        float t_min_global = 0;
        float t_max_global = 0;
        int parent_node_idx = 0;
        int current_node_idx = 0;

        //if(offset == 0)
        //{
        //    out[0] = points_sorted[1250 + points_sorted_step*2];
        //    out[1] = 99;
        //}
        if(x >=0 && x < cols && y >= 0 && y < rows && offset < rows*cols) // if within image
        //if(x == 100 && y == 100)
        //if(false)
        {
            color_out[offset] = 0; // black for miss
            depth_out[offset] = 0; // zero for miss

            // calculate ray vector and normalize
            float3 ray_vec;
            ray_vec.x = K_inv[0]*x + K_inv[1]*y + K_inv[2];
            ray_vec.y = K_inv[3]*x + K_inv[4]*y + K_inv[5];
            ray_vec.z = K_inv[6]*x + K_inv[7]*y + K_inv[8];


            if(ray_vec.x == 0.0)
                ray_vec.x = 1.0/100000;
            if(ray_vec.y == 0.0)
                ray_vec.y = 1.0/100000;
            if(ray_vec.z == 0.0)
                ray_vec.z = 1.0/100000;

            float norm = ray_vec.x*ray_vec.x + ray_vec.y*ray_vec.y + ray_vec.z*ray_vec.z;
            norm = sqrt(norm);
            ray_vec.x = ray_vec.x/norm;
            ray_vec.y = ray_vec.y/norm;
            ray_vec.z = ray_vec.z/norm;

            // check if ray vector are all positive and set ray mask
            int ray_dir = 0;
            if(ray_vec.x < 0)
            {
                ray_dir = ray_dir | 1;
            }
            if(ray_vec.y < 0)
            {
                ray_dir = ray_dir | 2;
            }
            if(ray_vec.z < 0)
            {
                ray_dir = ray_dir | 4;
            }

            // get root node	
            int node_idx = paths[level] >> 8;
            int code = codes[node_idx];

            // calculate overall bounding box
            float3 node_minp = minp;
            float3 node_maxp = maxp;
            pcl::device::calcBoundingBox(level, code, node_minp, node_maxp);

            // calculate t value
            float3 t_0;
            float3 t_1;
            float t_enter;
            float t_exit;
            pcl::device::calcRayPara(ray_vec, node_minp, node_maxp, t_0, t_1, t_enter, t_exit);

            // calculate smallest cell size to determine whether ray hit the point
            node_minp = minp;
            node_maxp = maxp;
            pcl::device::calcBoundingBox(resolution, code, node_minp, node_maxp); //15
            float3 smallest_cell_size;
            pcl::device::calcCellSize(node_minp, node_maxp, smallest_cell_size);

            // set stack for root
            parent[level] = 0; // -1 mean root
            t_min_global = t_enter;
            t_max_global = t_exit;
            t_min[level] = t_min_global;
            t_max[level] = t_max_global;

            // get first child idx
            int child_idx = pcl::device::getChildIdx(t_0, t_1, ray_vec);
            idx[level] = -1; // for root there is no sibiling

            // get first child code
            current_node_idx = node_idx;
            code = codes[current_node_idx];
            code = ((code << 3) | (child_idx ^ ray_dir));

            // increase level
            level++;
            parent_node_idx = current_node_idx;

            bool ifFound = false;
            bool ifMiss = false;
            //while loop
            while(!ifFound && !ifMiss)
            {
                if(level >= 11)
                {
                	color_out[offset] = 0; // black for miss
            		depth_out[offset] = 0; // zero for miss
                }

                /*******************intersection*******************/
                // calculate bounding box 
                node_minp = minp;
                node_maxp = maxp;
                pcl::device::calcBoundingBox(level, code, node_minp, node_maxp);

                // calculate t value
                pcl::device::calcRayPara(ray_vec, node_minp, node_maxp, t_0, t_1, t_enter, t_exit);

                // check child mask to see if target child exist
                int children_mask = nodes[parent_node_idx] & 0xFF;
                int child_node_idx  = nodes[parent_node_idx] >> 8;
                int child_offset = pcl::device::getChildOffset(children_mask, (child_idx ^ ray_dir));

                /**********************push**************************/
                if(child_offset >= 0)
                {
                    // child exist
                    if((t_min_global < t_max_global) && (t_enter < t_exit))
                    {
                        // push into stack
                        parent[level] = parent_node_idx;
                        t_min[level] = t_enter;
                        t_max[level] = t_max_global; //??
                        idx[level] = child_idx; // idx for current level

                        current_node_idx = child_node_idx + child_offset;
                        children_mask = nodes[current_node_idx] & 0xFF;
                        bool isLeaf = (children_mask == 0);
                        if(isLeaf)
                        {
                            //if(toPrint)
                            //{
                            //    cout << "\n******Found leaf******" << endl;
                            //    cout << "Leaf node is " << current_node_idx << endl;
                            //}
                            int idx_begin = begs[current_node_idx];
                            int idx_end = ends[current_node_idx];

                            //int step = points_sorted_step;
                            for(int i = idx_begin; i < idx_end; i++)
                            {
                                int idx =  indices[i];
                                float x =  points_sorted[i];
                                float y =  points_sorted[i + points_sorted_step];
                                float z =  points_sorted[i + points_sorted_step*2];
                                if(pcl::device::ifWithinSmallestCell(smallest_cell_size, ray_vec, x, y, z))
                                {

                                    uint32_t rgb =  color[idx];
            						uint8_t r = (rgb >> 16) & 0x0000ff;
            						uint8_t g = (rgb >> 8)  & 0x0000ff;
            						uint8_t b = (rgb)       & 0x0000ff;
            						int gray = r*0.299f + g*0.587f + b*0.114f;
                                    color_out[offset] = gray;
                                    if(z < 0)
                                    {
                                    	depth_out[offset] = 0; // zero for error
                                    }
                                    else
                                    {
                                    	depth_out[offset] = (unsigned int)10*z; // scale factor
                                    }                              
                                                             
                                    ifFound = true;
                                    return;
                                }

                            }
                        }
                        else
                        {
                            child_idx = pcl::device::getChildIdx(t_0, t_1, ray_vec);
                            code = codes[current_node_idx];
                            code = ((code << 3) | (child_idx ^ ray_dir));

                            // increase level
                            level++;
                            parent_node_idx = current_node_idx;

                            continue; // continue while loop
                        }

                    }
                }

                /**********************advance**************************/
                bool ifExitCurrentLevel = pcl::device::getNextNodeIdx(child_idx, t_1);
                t_min_global = t_exit;

                /**********************pop**************************/
                while(ifExitCurrentLevel)
                {
                    if(level <= 1) //?? <= or <
                    {
                        ifMiss = true;
                        ifExitCurrentLevel = false;
                        return;
                    }
                    else
                    {
                        level--;
                        // pop out
                        parent_node_idx = parent[level];
                        child_idx = idx[level];

                        // get previous voxel
                        code = codes[parent_node_idx];
                        code = ((code << 3) | (child_idx ^ ray_dir));

                        t_max_global = t_max[level];

                        node_minp = minp;
                        node_maxp = maxp;
                        pcl::device::calcBoundingBox(level, code, node_minp, node_maxp);

                        pcl::device::calcRayPara(ray_vec, node_minp, node_maxp, t_0, t_1, t_enter, t_exit);

                        ifExitCurrentLevel = pcl::device::getNextNodeIdx(child_idx, t_1);
                        t_min_global = t_exit;
                    }

                }
                if(!ifMiss)
                {
                    // get new voxel
                    code = codes[parent_node_idx];
                    code = ((code << 3) | (child_idx ^ ray_dir));
                }  
            }// end while loop
        }// end if

    } 
} 
}
}

void pcl::device::OctreeImpl::getRaytracingImage(int rows, int cols, unsigned char* color_out, unsigned short* depth_out, float* K_inv, int resolution) const
{
    cudaEvent_t start_cuda, stop_cuda;
    float time;
    cudaEventCreate(&start_cuda);
    cudaEventCreate(&stop_cuda);

	// init
	unsigned char* color_out_device;
	unsigned short* depth_out_device;
	int N = rows*cols;
	cudaSafeCall( cudaMalloc((void**)&color_out_device, N*sizeof(unsigned char)) );
	cudaSafeCall( cudaMalloc((void**)&depth_out_device, N*sizeof(unsigned short)) );
	
	float* K_inv_device; 
	cudaSafeCall( cudaMalloc((void**)&K_inv_device, 9*sizeof(float)) );
	cudaSafeCall( cudaMemcpy(K_inv_device, K_inv, 9*sizeof(float), cudaMemcpyHostToDevice) );
    

    int threadx = 8;
    int thready = 8;
    dim3 blocks((cols + threadx - 1)/threadx, (rows + thready -1)/thready);
    dim3 threads(threadx,thready);

	cudaEventRecord(start_cuda, 0);

    // kernal function call
    pcl::device::raytracing::Kernelraytracing<<<blocks, threads>>>(rows, cols, color_out_device, depth_out_device,
      octreeGlobal.nodes, octreeGlobal.codes, 
      octreeGlobal.begs, octreeGlobal.ends, 
      octreeGlobal.minp, octreeGlobal.maxp,
       points_sorted.ptr(), indices.ptr(), color.ptr(),
       points_sorted.elem_step() , K_inv_device, resolution);

    cudaEventRecord(stop_cuda, 0);
    cudaEventSynchronize(stop_cuda);
    cudaEventElapsedTime(&time, start_cuda, stop_cuda);
    printf("Raytracing take %f ms.\n", time);

    cudaEventRecord(start_cuda, 0);
	
	// copy back
    cudaSafeCall( cudaMemcpy(color_out, color_out_device, N*sizeof(unsigned char), cudaMemcpyDeviceToHost) ); 
	cudaSafeCall( cudaMemcpy(depth_out, depth_out_device, N*sizeof(unsigned short), cudaMemcpyDeviceToHost) ); 

    cudaEventRecord(stop_cuda, 0);
    cudaEventSynchronize(stop_cuda);
    cudaEventElapsedTime(&time, start_cuda, stop_cuda);
    printf("Copy back take %f ms.\n", time);

}
