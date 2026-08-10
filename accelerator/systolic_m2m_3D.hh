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


#ifdef NANO_3D

#include <cstddef>
#include <cstdint>

#define KERNEL_DIM SA_SIZE
#define W_DATA 4

#define MAX_COL (SA_H/W_DATA)
#define MAX_COL_OUT (SA_W/W_DATA)


#define mem2d(data,data_len,row,col)   data[((row)*(data_len))+(col)]

class SystolicMatrixMultiplication_3D {
  private:
    // System this ACM belongs to.

    // Loops through the output to select which one to extract
    int output_cnt = 0;

    int8_t weights[N_3D_LAYERS][SA_H * SA_W]{};
    int8_t inWaitingMemory[N_3D_LAYERS][SA_H * SA_H]{};
    int8_t inputMemory[N_3D_LAYERS][SA_H * SA_W]{};
    uint8_t outWaitingMemory[N_3D_LAYERS][SA_W * SA_W]{};
    int32_t outputMemory[N_3D_LAYERS][SA_W * (SA_H + 1)]{};

    bool non_zero_tile = false;
    
  public:
    bool loadWeights(int idx, uint32_t  val, int layer);
    uint32_t inputQueue_3Dnano(int col, uint32_t  val, int out_cnt);
    void printWeights();
    uint32_t streamInOut_3Dnano(uint32_t val, int out_cnt);
 };


#endif // NANO_3D

#endif // __SYSTOLIC_M2M_3D_H__
