/*⭐⭐⭐******************************************************************⭐⭐⭐*
 * Authors      :    Chen Feng <cfengag at connect dot ust dot hk>, UAV Group, ECE, HKUST.
 *                   Guiyong Zheng <shverses at gmail dot com>, SAI, SYSU.
 * Homepage     :    https://chen-albert-feng.github.io/AlbertFeng.github.io/
 *                   https://gy920.github.io/
 * Date         :    Apr. 2024
 * E-mail       :    cfengag at connect dot ust dot hk
 *                   shverses at gmail dot com
 * Description  :    This file is the header file of Extra_Del class, which implements the
 *                   advanced index retrieval of Eigen::Matrix and Eigen::Vector.
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

#ifndef _EXTRA_DEL_H_
#define _EXTRA_DEL_H_

#include<iostream>
#include<algorithm>
#include<Eigen/Eigen>

using namespace std;
using namespace Eigen;

namespace flyco
{

class Extra_Del
{
public:
	MatrixXd rows_ext_V(VectorXi ind, MatrixXd matrix);
	MatrixXd rows_ext_M(MatrixXd ind, MatrixXd matrix);
	MatrixXd cols_ext_V(VectorXi ind, MatrixXd matrix);
	MatrixXd cols_ext_M(MatrixXd ind, MatrixXd matrix);
	MatrixXd rows_del_M(MatrixXd ind, MatrixXd matrix);
	MatrixXd cols_del_M(MatrixXd ind, MatrixXd matrix);
};

}

#endif