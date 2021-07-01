
/*
 * Copyright (c) 2021, NVIDIA CORPORATION. All rights reserved.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a
 * copy of this software and associated documentation files (the "Software"),
 * to deal in the Software without restriction, including without limitation
 * the rights to use, copy, modify, merge, publish, distribute, sublicense,
 * and/or sell copies of the Software, and to permit persons to whom the
 * Software is furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL
 * THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
 * DEALINGS IN THE SOFTWARE.
 */

#include "contrib/nlohmann/json/json.hpp"

#include <cstdlib>
#include <fstream>
#pragma once

using json = nlohmann::json;

namespace cudnn_frontend {

// Loads the json handle from the json file 
// json file is defined by environment variable
// CUDNN_ERRATA_JSON_FILE. If the environment variable
// is not set the value set in the API is considered.
static bool
load_from_config(json &json_handle, const std::string & errata_json) {
    const char * err_json = std::getenv("CUDNN_ERRATA_JSON_FILE");
    if (err_json == NULL && errata_json == "") {return false;}
    if (err_json == NULL) { err_json = errata_json.c_str();}
    std::ifstream ifs(err_json, std::ifstream::in);
    if (!ifs.is_open() || !ifs.good()) {return false;}
    ifs >> json_handle;
    return true;
}

template <typename T>
static bool 
check_rule(const json &json_handle, const std::string & executionPlanTag,
    cudnnHandle_t handle, T fn) {
    std::string operation = json_handle["operation"];
    int64_t engine        =  json_handle["engine"];
    uint64_t cudnn_start     =  0;
    uint64_t cudnn_end       =  -1;
    if (json_handle.contains("cudnn_version_start")) {
        cudnn_start   =  json_handle["cudnn_version_start"];
    }
    if (json_handle.contains("cudnn_version_end")) {
        cudnn_end     =  json_handle["cudnn_version_end"];
    }
    std::string tag_prefix = operation + "_eng" + std::to_string(engine) + "_"; 
    std::string mod_tag    = executionPlanTag + "_";
    bool blocked = 
        std::equal(tag_prefix.begin(), tag_prefix.end(), mod_tag.begin()) &&
        CUDNN_VERSION >= cudnn_start &&
        CUDNN_VERSION < cudnn_end;

    if (blocked && json_handle.contains("knob")) { // Short circuit if operation and engine do not match
        for (auto& kv : json_handle["knob"]) {
            blocked = blocked &&
                (executionPlanTag.find(kv) != std::string::npos);
        }
    }

    blocked = blocked && fn(); 
    return blocked;

    (void) handle;
}

// Takes in an initialzed json handle and checks if it satisfies the 
// condition for running it. Returns true if the given executionPlanTag
// is faulty.
template <typename T>
static bool
check_errata(const json &json_handle, const std::string & executionPlanTag,
    cudnnHandle_t handle, T fn) {

    for (auto const &rule : json_handle["rules"]) {
        if (check_rule<T>(rule, executionPlanTag, handle, fn)) {
            return true;
        }
    }

    return false;
}

}
