/*⭐⭐⭐******************************************************************⭐⭐⭐*
 * Authors      :    Chen Feng <cfengag at connect dot ust dot hk>, UAV Group, ECE, HKUST.
 *                   Guiyong Zheng <shverses at gmail dot com>, SAI, SYSU.
 * Homepage     :    https://chen-albert-feng.github.io/AlbertFeng.github.io/
 *                   https://gy920.github.io/
 * Date         :    Sept. 2024
 * E-mail       :    cfengag at connect dot ust dot hk
 *                   shverses at gmail dot com
 * Description  :    This file implements common math utilities for FlyCo simulation.
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

namespace math_common
{
    template <typename T>
    inline T rad2deg(const T radians)
    {
        return (radians / M_PI) * 180.0;
    }

    template <typename T>
    inline T deg2rad(const T degrees)
    {
        return (degrees / 180.0) * M_PI;
    }

    template <typename T>
    inline T wrap_to_pi(T radians)
    {
        int m = (int) ( radians/ (2*M_PI) );
        radians = radians - m*2*M_PI;
        if (radians > M_PI)
            radians -= 2.0*M_PI;
        else if (radians < -M_PI)
            radians += 2.0*M_PI;
        return radians;
    }

    template <typename T>
    inline void wrap_to_pi_inplace(T& a)
    {
        a = wrap_to_pi(a);
    }

    template <class T>
    inline T angular_dist(T from, T to)
    {
        wrap_to_pi_inplace(from);
        wrap_to_pi_inplace(to);
        T d = to - from;
        if (d > M_PI)
            d -= 2. * M_PI;
        else if (d < -M_PI)
            d += 2. * M_PI;
        return d;
    }
}