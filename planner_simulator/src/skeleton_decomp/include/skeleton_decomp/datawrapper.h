/*⭐⭐⭐******************************************************************⭐⭐⭐*
 * Authors      :    Chen Feng <cfengag at connect dot ust dot hk>, UAV Group, ECE, HKUST.
 *                   Guiyong Zheng <shverses at gmail dot com>, SAI, SYSU.
 * Homepage     :    https://chen-albert-feng.github.io/AlbertFeng.github.io/
 *                   https://gy920.github.io/
 * Date         :    Apr. 2024
 * E-mail       :    cfengag at connect dot ust dot hk
 *                   shverses at gmail dot com
 * Description  :    This file is the data wrapper structure for plane query in FlyCo.
 * Copyright    :    Copyright (c) 2026 Chen Feng and Guiyong Zheng.
 * License      :    PolyForm Noncommercial License 1.0.0
 *                   <https://polyformproject.org/licenses/noncommercial/1.0.0/>
 *
 *                   This software is released for noncommercial research and
 *                   educational use only. You may use, modify, and distribute
 *                   this software for noncommercial purposes, subject to the
 *                   terms of the PolyForm Noncommercial License 1.0.0.
 *
 *                   Commercial use, including use in commercial products,
 *                   commercial services, paid consulting, or internal business
 *                   operations, is prohibited without prior written permission
 *                   from the copyright holders.
 *
 *                   This software is provided "as is", without warranty of any
 *                   kind, express or implied.
 * Project      :    FlyCo: Foundation Model-Empowered Drones for Autonomous 3D Structure Scanning in Open-World Environments
 * Website      :    https://hkust-aerial-robotics.github.io/FC-Planner/
 *⭐⭐⭐*****************************************************************⭐⭐⭐*/

#ifndef _DATAWRAPPER_H_
#define _DATAWRAPPER_H_

#include <assert.h>
#include <vector>
#include <iostream>
#include <math.h>

using namespace std;

namespace flyco
{

class DataWrapper{
    private: 
    double*   	     data;
    int       		 npoints;
    const static int ndim = 3; 
        
    public:
        
    void factory( double* data, int npoints ){
        this->data 	  = data;
        this->npoints = npoints;
    }
    /** 
     *  Data retrieval function
     *  @param a address over npoints
     *  @param b address over the dimensions
     */
    inline double operator()(int a, int b){
        assert( a<npoints );
        assert( b<ndim );
        return data[ a + npoints*b ];
    }
    // retrieve a single point at offset a, in a vector (preallocated structure)
    inline void operator()(int a, vector<double>& p){
        assert( a<npoints );
        assert( (int)p.size() == ndim );
        p[0] = data[ a + 0*npoints ];
        p[1] = data[ a + 1*npoints ];
        p[2] = data[ a + 2*npoints ];
    }
    int length(){
        return this->npoints;
    }
};

}

#endif