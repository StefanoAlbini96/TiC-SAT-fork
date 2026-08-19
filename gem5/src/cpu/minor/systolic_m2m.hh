/*
 * Copyright (c) 2020 EPFL
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
 * Authors: Alireza Amirshahi
 */

#ifndef __SYSTOLIC_M2M_H__
#define __SYSTOLIC_M2M_H__


#include "params/SystolicMatrixMultiplication.hh"

#include "sim/sim_object.hh"

#include "cpu/minor/cpu.hh"
#include "cpu/minor/func_unit.hh"
#include "base/types.hh"
#include <iostream>
#include <numeric>
#include <vector>


#include <cstddef>
#include <cstdint>


namespace gem5 {


// #define MAX_COL (SA_SIZE/W_DATA)
// #define MAX_COL_OUT (MAX_COL)

#define mem2d(data,data_len,row,col)   data[((row)*(data_len))+(col)]

class SystolicMatrixMultiplication  : public SimObject {
  private:

    const int kernelDim;
    const int w_data = 4;

    const int max_col = kernelDim / w_data;
    const int max_col_out = max_col;


    std::vector<int8_t>   weights;
    std::vector<int8_t>   inWaitingMemory;
    std::vector<int8_t>   inputMemory;
    std::vector<uint8_t>  outWaitingMemory;
    std::vector<int32_t>  outputMemory;



    bool non_zero_tile = false;
    
  public:
    
    SystolicMatrixMultiplication(const SystolicMatrixMultiplicationParams &p);



    bool      loadWeights(int idx, uint32_t  val);
    uint32_t  inputQueue(int col, uint32_t  val);
    uint32_t  inputQueue_3Dnano(int col, uint32_t  val);
    void      printWeights();
    uint32_t  streamInOut(uint32_t val);
    uint32_t  streamInOut_3Dnano(uint32_t val);
 };

}

#endif // __SYSTOLIC_M2M_H__
