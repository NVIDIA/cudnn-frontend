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

#pragma once

#include <iostream>
#include <fstream>
#include <cstring>

#include "cudnn_backend_base.h"
namespace  cudnn_frontend {

static const char *
get_environment(const char *name) {
#ifdef WIN32
#pragma warning(disable:4996)
#define  _CRT_SECURE_NO_WARNINGS
#endif

    return std::getenv(name);
}

inline bool &
isLoggingEnabled() {
    static bool log_enabled = get_environment("CUDNN_FRONTEND_LOG_INFO") && std::strncmp(get_environment("CUDNN_FRONTEND_LOG_INFO"), "0",1);
    return log_enabled;
}
 
inline std::ostream &
getStream() {                                                                                                                                                                                                                      
    static std::ofstream outFile;
    static std::ostream & stream  = get_environment("CUDNN_FRONTEND_LOG_FILE")
         ?  (std::strncmp(get_environment("CUDNN_FRONTEND_LOG_FILE"), "stdout", 6) == 0 
             ? std::cout : (std::strncmp(get_environment("CUDNN_FRONTEND_LOG_FILE"), "stderr", 6) == 0 
                 ? std::cerr : (outFile.open(get_environment("CUDNN_FRONTEND_LOG_FILE"), std::ios::out), outFile)))
         : (isLoggingEnabled() = false, std::cout);
    return stream;
}


class ConditionalStreamer {
  private:
    std::ostream &stream;
  public:
    ConditionalStreamer(std::ostream &stream_) : stream(stream_){}
  
    template <typename T>
    const ConditionalStreamer &
    operator<< (const T &t) const {
        if (isLoggingEnabled()) {stream << t;}  
        return *this;
    }
  
    const ConditionalStreamer &
    operator<< (std::ostream &(*spl)(std::ostream &)) const {
        if (isLoggingEnabled()) {stream << spl;}
        return *this;
    }
};


inline ConditionalStreamer &
getLogger() {                                         
    static ConditionalStreamer opt(getStream());
    return opt;
}                    

static
std::ostream &
operator << (std::ostream &os, const BackendDescriptor & desc) {
    if (isLoggingEnabled()) {os << desc.describe();}
    return os;
}
}
