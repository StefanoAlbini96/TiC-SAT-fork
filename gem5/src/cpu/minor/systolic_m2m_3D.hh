/*
 * Copyright (c) 2026 EPFL
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are
 * met: redistributions of source code must retain the above copyright
 * notice, this list of conditions and the following disclaimer;
 * redistributions in binary form must reproduce the above copyright
 * notice, this list of conditions and the following disclaimer in the
 * documentation and/or other materials provided with the distribution;
 * neither the name of the copyright holders nor the names of its
 * contributors may be used to endorse or promote products derived from
 * this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
 * "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
 * LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
 * A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
 * OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
 * SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
 * LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
 * DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
 * THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 * Authors: Stefano Albini
 */

#ifndef __SYSTOLIC_M2M_3D_H__
#define __SYSTOLIC_M2M_3D_H__


#include "params/SystolicMatrixMultiplication_3D.hh"

#include "sim/sim_object.hh"

#include "cpu/minor/cpu.hh"
#include "cpu/minor/func_unit.hh"
#include "base/types.hh"
#include <iostream>
#include <numeric>
#include <vector>


#include <cstddef>
#include <cstdint>

// #define KERNEL_DIM SA_SIZE
// #define W_DATA 4

// #define MAX_COL (SA_H/W_DATA)
// #define MAX_COL_OUT (SA_W/W_DATA)

namespace gem5
{

#define mem2d(data,data_len,row,col)   data[((row)*(data_len))+(col)]

class SystolicMatrixMultiplication_3D  : public SimObject {
  private:
  
    // Height and width of the Systolic Array
    const int kernelDim_h;
    const int kernelDim_w;

    const int n_3D_layers;

    const int w_data = 4;

    const int max_col = kernelDim_h / w_data;
    const int max_col_out = kernelDim_w / w_data;


    // Loops through the output to select which one to extract
    int output_cnt = 0;

    std::vector<std::vector<int8_t>> weights;
    std::vector<std::vector<int8_t>> inWaitingMemory;
    std::vector<std::vector<int8_t>> inputMemory;
    std::vector<std::vector<uint8_t>> outWaitingMemory;
    std::vector<std::vector<int32_t>> outputMemory;

    // std::vector<int8_t> weights[n_3D_layers][kernelDim_h * kernelDim_w]{};
    // std::vector<int8_t> inWaitingMemory[n_3D_layers][kernelDim_h * kernelDim_h]{};
    // std::vector<int8_t> inputMemory[n_3D_layers][kernelDim_h * kernelDim_w]{};
    // std::vector<uint8_t> outWaitingMemory[n_3D_layers][kernelDim_w * kernelDim_w]{};
    // std::vector<int32_t> outputMemory[n_3D_layers][kernelDim_w * (kernelDim_h + 1)]{};

    bool non_zero_tile = false;
    
  public:

    SystolicMatrixMultiplication_3D(const SystolicMatrixMultiplication_3DParams &p);

    bool loadWeights(int idx, uint32_t  val, int layer);
    uint32_t inputQueue_3Dnano(int col, uint32_t  val, int out_cnt);
    void printWeights();
    uint32_t streamInOut_3Dnano(uint32_t val, int out_cnt);
 };

}

#endif // __SYSTOLIC_M2M_3D_H__
