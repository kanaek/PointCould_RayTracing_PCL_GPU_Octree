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
 *  Author: Anatoly Baskeheev, Itseez Ltd, (myname.mysurname@mycompany.com)
 */

#ifndef _PCL_GPU_OCTREE_BOXUTILS_HPP_
#define _PCL_GPU_OCTREE_BOXUTILS_HPP_

#include "utils/morton.hpp"

namespace pcl
{
    namespace device
    {
        __device__ __host__ __forceinline__
        static bool checkIfNodeInsideSphere(const float3& minp, const float3& maxp, const float3& c, float r)
        {
            r *= r;

            float d2_xmin = (minp.x - c.x) * (minp.x - c.x);
            float d2_ymin = (minp.y - c.y) * (minp.y - c.y);
            float d2_zmin = (minp.z - c.z) * (minp.z - c.z);

            if (d2_xmin + d2_ymin + d2_zmin > r)
                return false;

            float d2_zmax = (maxp.z - c.z) * (maxp.z - c.z);

            if (d2_xmin + d2_ymin + d2_zmax > r)
                return false;

            float d2_ymax = (maxp.y - c.y) * (maxp.y - c.y);

            if (d2_xmin + d2_ymax + d2_zmin > r)
                return false;

            if (d2_xmin + d2_ymax + d2_zmax > r)
                return false;

            float d2_xmax = (maxp.x - c.x) * (maxp.x - c.x);

            if (d2_xmax + d2_ymin + d2_zmin > r)
                return false;

            if (d2_xmax + d2_ymin + d2_zmax > r)
                return false;

            if (d2_xmax + d2_ymax + d2_zmin > r)
                return false;

            if (d2_xmax + d2_ymax + d2_zmax > r)
                return false;

            return true;
        }

        __device__ __host__ __forceinline__
        static bool checkIfNodeOutsideSphere(const float3& minp, const float3& maxp, const float3& c, float r)
        {
            if (maxp.x < (c.x - r) ||  maxp.y < (c.y - r) || maxp.z < (c.z - r))
                return true;

            if ((c.x + r) < minp.x || (c.y + r) < minp.y || (c.z + r) < minp.z)
                return true;

            return false;
        }

        __device__ __host__ __forceinline__
        static void calcBoundingBox(int level, int code, float3& res_minp, float3& res_maxp)
        {        
            int cell_x, cell_y, cell_z;
            Morton::decomposeCode(code, cell_x, cell_y, cell_z);   

            float cell_size_x = (res_maxp.x - res_minp.x) / (1 << level);
            float cell_size_y = (res_maxp.y - res_minp.y) / (1 << level);
            float cell_size_z = (res_maxp.z - res_minp.z) / (1 << level);

            res_minp.x += cell_x * cell_size_x;
            res_minp.y += cell_y * cell_size_y;
            res_minp.z += cell_z * cell_size_z;

            res_maxp.x = res_minp.x + cell_size_x;
            res_maxp.y = res_minp.y + cell_size_y;
            res_maxp.z = res_minp.z + cell_size_z;       
        }

        __device__ __host__ __forceinline__
        static void calcCellSize(float3& res_minp, float3& res_maxp, float3& cell_size)
        {
            float cell_size_x = res_maxp.x - res_minp.x;
            float cell_size_y = res_maxp.y - res_minp.y;
            float cell_size_z = res_maxp.z - res_minp.z;

            cell_size.x = cell_size_x;
            cell_size.y = cell_size_y;
            cell_size.z = cell_size_z;
        }

        __device__ __host__ __forceinline__
        static bool checkIfNodeOutsideCell(const float3& minp, const float3& maxp, const float3& c)
        {
            if (maxp.x < c.x ||  maxp.y < c.y || maxp.z < c.z)
                return true;

            if (c.x < minp.x || c.y < minp.y || c.z < minp.z)
                return true;

            return false;
        }

        __device__ __host__ __forceinline__
        static int getChildCount(int interger)
        {
            int count = 0;
            while(interger > 0)
            {
                if (interger & 1)
                    ++count;
                interger>>=1;
            }
            return count;
        }

        __device__ __host__ __forceinline__
        static int getChildOffset(int mask, int idx)
        {
            int offset = -1;
            for(int i = 0; i<= idx; i++)
            {
                int temp = (mask >> i);

                if(i != idx)
                {
                    if(temp & 1)
                    {
                        offset++;
                    }
                }
                else
                {
                    if(temp & 1)
                    {
                        offset++;
                    }
                    else
                    {
                        offset = -1;
                    }
                }
            }

            // if not exist return -1
            return offset;
        }

        __device__ __host__ __forceinline__
        static void calcRayPara(float3 ray_vec, float3 & node_minp, float3 & node_maxp, float3 & t_0, float3 & t_1, float & t_enter, float & t_exit)
        {
            float3 cell_size;
            cell_size.x = node_maxp.x - node_minp.x;
            cell_size.y = node_maxp.y - node_minp.y;
            cell_size.z = node_maxp.z - node_minp.z;

            float3 o_m;
            o_m.x = node_minp.x + cell_size.x/2.0;
            o_m.y = node_minp.y + cell_size.y/2.0;
            o_m.z = node_minp.z + cell_size.z/2.0;

            float3 ray_origin;
            if(ray_vec.x > 0)
            {
                ray_origin.x = 0.0;
            }
            else
            {
                ray_origin.x = o_m.x*(2.0);
                ray_vec.x = ray_vec.x*(-1.0);
            }
            if(ray_vec.y > 0)
            {
                ray_origin.y = 0.0;
            }
            else
            {
                ray_origin.y = o_m.y*(2.0);
                ray_vec.y = ray_vec.y*(-1.0);
            }
            if(ray_vec.z > 0)
            {
                ray_origin.z = 0.0;
            }
            else
            {
                ray_origin.z = o_m.z*(2.0);
                ray_vec.z = ray_vec.z*(-1.0);
            }

            t_0.x = (node_minp.x - ray_origin.x)/ray_vec.x;
            t_0.y = (node_minp.y - ray_origin.y)/ray_vec.y;
            t_0.z = (node_minp.z - ray_origin.z)/ray_vec.z;

            t_1.x = (node_maxp.x - ray_origin.x)/ray_vec.x;
            t_1.y = (node_maxp.y - ray_origin.y)/ray_vec.y;
            t_1.z = (node_maxp.z - ray_origin.z)/ray_vec.z;

            t_enter = t_0.x;
            if(t_0.y > t_enter)
                t_enter = t_0.y;
            if(t_0.z > t_enter)
                t_enter = t_0.z;

            t_exit = t_1.x;
            if(t_1.y < t_exit)
                t_exit = t_1.y;
            if(t_1.z < t_exit)
                t_exit = t_1.z;
        }

        __device__ __host__ __forceinline__
        static int getChildIdx(float3 & t_0, float3 & t_1, float3 & ray_vec)
        {
            int idx = 0;
            float3 t_m;
            t_m.x = (t_0.x + t_1.x)/2.0;
            t_m.y = (t_0.y + t_1.y)/2.0;
            t_m.z = (t_0.z + t_1.z)/2.0;

            float t_enter = t_0.x;
            if(t_0.y > t_enter)
                t_enter = t_0.y;
            if(t_0.z > t_enter)
                t_enter = t_0.z;

            if(t_enter > t_m.x)
                idx = (idx | 1);

            if(t_enter > t_m.y)
                idx = (idx | 2);

            if(t_enter > t_m.z)
                idx = (idx | 4);

            return idx;
        }

        __device__ __host__ __forceinline__
        static bool getNextNodeIdx(int & idx, float3 & t_1)
        {
            bool ifExitCurrentLevel = false;

            float t_exit = t_1.x;
            if(t_1.y < t_exit)
                t_exit = t_1.y;
            if(t_1.z < t_exit)
                t_exit = t_1.z;

            if(t_exit == t_1.x)
            {
                if(((idx >> 0) & 1) == 1)
                    ifExitCurrentLevel = true;

                idx = idx | 1;
            }
            else if(t_exit == t_1.y)
            {
                if(((idx >> 1) & 1) == 1)
                    ifExitCurrentLevel = true;

                idx = idx | 2;
            }
            else
            {
                if(((idx >> 2) & 1) == 1)
                    ifExitCurrentLevel = true;

                idx = idx | 4;
            }

            return ifExitCurrentLevel;
        }

        __device__ __host__ __forceinline__
        static bool ifWithinSmallestCell(float3 & cell_size, float3 & point, float x, float y, float z)
        {
            // point is the end point of the ray
            float squared_norm = point.x*point.x + point.y*point.y + point.z*point.z;

            float t = (-x*point.x) +  (-y*point.y) +  (-z*point.z);
            t = t*(-1);
            t = t/squared_norm;

            float part1 = (-x + point.x*t);
            float part2 = (-y + point.y*t);
            float part3 = (-z + point.z*t);
            part1 = part1*part1;
            part2 = part2*part2;
            part3 = part3*part3;

            float distance = part1 + part2 + part3;
            float cell_diameter = cell_size.x*cell_size.x +  cell_size.y*cell_size.y +  cell_size.z*cell_size.z;

            if(distance <= cell_diameter)
                return true;

            return false;
        }

    }
}

#endif /* _PCL_GPU_OCTREE_BOXUTILS_HPP_ */
