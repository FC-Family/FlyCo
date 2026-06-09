/*⭐⭐⭐******************************************************************⭐⭐⭐*
 * Authors      :    Chen Feng <cfengag at connect dot ust dot hk>, UAV Group, ECE, HKUST.
 *                   Guiyong Zheng <shverses at gmail dot com>, SAI, SYSU.
 * Homepage     :    https://chen-albert-feng.github.io/AlbertFeng.github.io/
 *                   https://gy920.github.io/
 * Date         :    Jun. 2024
 * E-mail       :    cfengag at connect dot ust dot hk
 *                   shverses at gmail dot com
 * Description  :    This file is the logger information header file.
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

#ifndef _FLYCO_PLANNER_MANAGER_LOGGER_HPP_
#define _FLYCO_PLANNER_MANAGER_LOGGER_HPP_

#define ANSI_COLOR_YELLOW_BOLD "\033[1;33m"
#define ANSI_COLOR_GREEN_BOLD "\033[1;32m"

#include <iostream>
#include <fstream>
#include <sstream>
#include <array>
#include <string>
#include <chrono>
#include <ctime>
#include <stdexcept>
#include <cpuid.h>
#include <cstdio>
#include <memory>
#include <iomanip>
#include <sys/utsname.h>
#include <regex>
#include <Eigen/Core>
#include <pcl/pcl_config.h>
#include <boost/version.hpp>

namespace flyco
{
namespace logger
{
    inline std::string get_current_time()
    {
        auto now = std::chrono::system_clock::now();
        auto now_c = std::chrono::system_clock::to_time_t(now);
        return std::ctime(&now_c);
    }
    
    inline std::string get_cpu_model()
    {
        std::array<unsigned int, 4> cpui;
        std::string cpu_info;

        for (int i = 0x80000002; i <= 0x80000004; ++i) {
            __get_cpuid(i, &cpui[0], &cpui[1], &cpui[2], &cpui[3]);
            cpu_info += std::string((char*)&cpui[0], 4);
            cpu_info += std::string((char*)&cpui[1], 4);
            cpu_info += std::string((char*)&cpui[2], 4);
            cpu_info += std::string((char*)&cpui[3], 4);
        }

        return cpu_info;
    }

    inline std::string get_memory_info() 
    {
        std::ifstream meminfo("/proc/meminfo");
        std::string line;
        double physical_mem = 0.0;
        double virtual_mem = 0.0;
        std::stringstream ss;

        while (std::getline(meminfo, line)) {
            if (line.find("MemTotal") != std::string::npos) {
                physical_mem = std::stod(line.substr(line.find(":") + 1)) / (1024 * 1024);
            }
            if (line.find("SwapTotal") != std::string::npos) {
                virtual_mem = std::stod(line.substr(line.find(":") + 1)) / (1024 * 1024);
            }
        }

        ss << std::fixed << std::setprecision(2) << physical_mem << "GB Physical Memory, " << virtual_mem << "GB Virtual Memory";
        return ss.str();
    }

    inline std::string get_os_info()
    {
        struct utsname buffer;
        if (uname(&buffer) != 0) {
            return "Unknown OS";
        }

        std::string os_info;
        os_info += buffer.sysname;
        os_info += " ";
        os_info += buffer.release;
        os_info += " (";
        os_info += buffer.machine;
        os_info += ")";

        return os_info;
    }

    inline std::string get_gcc_version()
    {
        std::array<char, 128> buffer;
        std::string result;

        std::unique_ptr<FILE, decltype(&pclose)> pipe(popen("g++ --version", "r"), pclose);
        if (!pipe) {
            throw std::runtime_error("popen() failed!");
        }
        while (fgets(buffer.data(), buffer.size(), pipe.get()) != nullptr) {
            result += buffer.data();
        }

        std::regex version_regex(R"((\d+\.\d+\.\d+))");
        std::smatch version_match;
        if (std::regex_search(result, version_match, version_regex) && version_match.size() > 1) {
            result = version_match.str(1);
        } else {
            result = "GCC version not found";
        }

        return result;
    }

    inline std::string get_eigen_version()
    { 
        return std::to_string(EIGEN_WORLD_VERSION) + "." + 
           std::to_string(EIGEN_MAJOR_VERSION) + "." + 
           std::to_string(EIGEN_MINOR_VERSION);
    }

    inline std::string get_pcl_version()
    {
        return std::to_string(PCL_VERSION / 100000) + "." + 
           std::to_string((PCL_VERSION / 100) % 1000) + "." + 
           std::to_string(PCL_VERSION % 100);
    }

    inline std::string get_boost_version()
    {
        return std::to_string(BOOST_VERSION / 100000) + "." + 
           std::to_string((BOOST_VERSION / 100) % 1000) + "." + 
           std::to_string(BOOST_VERSION % 100);
    }
    
    inline void deviceInfo()
    {
        std::cout << std::endl;
        std::cout << ANSI_COLOR_YELLOW_BOLD;
        std::cout << "------------------------YOUR DEVICE INFO------------------------" << std::endl;
        std::cout << "Project           : " << "FlyCo" << std::endl;
        std::cout << "Author            : " << "Chen Feng & Guiyong Zheng" << std::endl;
        std::cout << "Current Time      : " << get_current_time();
        std::cout << "CPU Info          : " << get_cpu_model() << std::endl;
        std::cout << "RAM Info          : " << get_memory_info() << std::endl;
        std::cout << "OS Info           : " << get_os_info() << std::endl;
        std::cout << "GCC Version       : " << get_gcc_version() << std::endl;
        std::cout << "Eigen Version     : " << get_eigen_version() << std::endl;
        std::cout << "PCL Version       : " << get_pcl_version() << std::endl;
        std::cout << "Boost Version     : " << get_boost_version() << std::endl;
        std::cout << "----------------------------------------------------------------" << std::endl;
        std::cout << std::endl;
    }
} // namespace logger
} // namespace flyco

#endif

