/*⭐⭐⭐******************************************************************⭐⭐⭐*
 * Authors      :    Chen Feng <cfengag at connect dot ust dot hk>, UAV Group, ECE, HKUST.
 *                   Guiyong Zheng <shverses at gmail dot com>, SAI, SYSU.
 * Homepage     :    https://chen-albert-feng.github.io/AlbertFeng.github.io/
 *                   https://gy920.github.io/
 * Date         :    Sept. 2024
 * E-mail       :    cfengag at connect dot ust dot hk
 *                   shverses at gmail dot com
 * Description  :    This file is the implementation file of AirSim settings parsing for FlyCo
 *                   simulation.
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

#include "airsim_settings_parser.h"

AirSimSettingsParser::AirSimSettingsParser()
{
    success_ = initializeSettings();
}

bool AirSimSettingsParser::success()
{
    return success_;
}

bool AirSimSettingsParser::readSettingsTextFromFile(std::string settingsFilepath, std::string& settingsText) 
{
    // check if path exists
    bool found = std::ifstream(settingsFilepath.c_str()).good(); 
    if (found)
    {
        std::ifstream ifs(settingsFilepath);
        std::stringstream buffer;
        buffer << ifs.rdbuf();
        // todo airsim's simhud.cpp does error checking here
        settingsText = buffer.str(); // todo convert to utf8 as done in simhud.cpp?
    }

    return found;
}

bool AirSimSettingsParser::getSettingsText(std::string& settingsText) 
{
    bool success = readSettingsTextFromFile(msr::airlib::Settings::Settings::getUserDirectoryFullPath("settings.json"), settingsText);
    return success;
}

std::string AirSimSettingsParser::getSimMode()
{
    Settings& settings_json = Settings::loadJSonString(settingsText_);
    return settings_json.getString("SimMode", "");
}

// mimics void ASimHUD::initializeSettings()
bool AirSimSettingsParser::initializeSettings()
{
    if (getSettingsText(settingsText_))
    {
        AirSimSettings::initializeSettings(settingsText_);

        // not sure where settings_json initialized in AirSimSettings::initializeSettings() is actually used
        Settings& settings_json = Settings::loadJSonString(settingsText_);
        std::string simmode_name = settings_json.getString("SimMode", "");
        std::cout << "simmode_name: " << simmode_name << std::endl; 

        AirSimSettings::singleton().load(std::bind(&AirSimSettingsParser::getSimMode, this));

        return true;
    }
    else
    {
        return false;
    }
}