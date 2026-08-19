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
// Constructor.


#include "cpu/minor/systolic_m2m.hh"

#include <cstdint>
#include <iostream>

#include "arch/arm/system.hh"


namespace gem5
{


SystolicMatrixMultiplication::SystolicMatrixMultiplication(const SystolicMatrixMultiplicationParams &p)
    :   
      SimObject(p),
      kernelDim(p.kernel_dim),
      weights(kernelDim * kernelDim, 0),
      inWaitingMemory(kernelDim * kernelDim, 0),
      inputMemory(kernelDim * kernelDim, 0),
      outWaitingMemory(kernelDim * kernelDim, 0),
      outputMemory(kernelDim * (kernelDim + 1), 0)
{

}


bool SystolicMatrixMultiplication::loadWeights(int idx, uint32_t val) {

    for (int i=0; i < w_data; i++){
        auto currVal = (int8_t)((val >> (8 * (w_data -i-1))) & 0xff);
        weights[idx + i] = currVal;
    }
    if (val!=0)
        non_zero_tile = true;
    return non_zero_tile;
}





uint32_t SystolicMatrixMultiplication::inputQueue(int col, uint32_t val) {
    // Split the input to an array
    for (int i=0; i < w_data; i++){
        auto currVal = (int8_t)((val >> (8 * (w_data - i -1))) & 0xff);
        int row_index = (col*w_data+i);
        mem2d(inWaitingMemory, kernelDim, row_index, kernelDim - row_index - 1) = currVal; // off-diagonal of the waiting memory
    }


    // printf("\n");
    // print_buf((int8_t*)outWaitingMemory, SA_W, SA_W);
    // printf("\n");

    // Return the output
    uint32_t result = 0;
    int result_idx = ((col+1)%max_col) * w_data;
    for (int i = 0; i < w_data; i++) {
        result |=  mem2d(outWaitingMemory, kernelDim, kernelDim - result_idx - i - 1, result_idx + i ) << (8 * (w_data - i - 1));
    }
    return result;
}

void SystolicMatrixMultiplication::printWeights() {
    std::cout << std::hex << (uint32_t) weights[0] << std::endl;
}









uint32_t SystolicMatrixMultiplication::streamInOut(uint32_t val) {
    non_zero_tile = false;
	int col = max_col - 1;
    // Split the input to an array
    // printf("New val --> ");
    for (int i=0; i < w_data; i++){
        auto currVal = (int8_t)((val >> (8 * (w_data - i -1))) & 0xff);
        // printf("%d, ", currVal);
        int row_index = (col*w_data+i);
        mem2d(inWaitingMemory, kernelDim, row_index, kernelDim - row_index - 1) = currVal; // off-diagonal of the waiting memory
    }
    // printf("\n");

    // Shift the waiting memory to the right for skewing
    for (int i = 0; i < kernelDim; i++) {
        mem2d(inputMemory, kernelDim, i, 0) = mem2d(inWaitingMemory, kernelDim, i, kernelDim - 1);
        for (int j= kernelDim - 1; j > 0; j--){ // TODO: shift only the right-hand triangle
            mem2d(inWaitingMemory, kernelDim, i, j) = mem2d(inWaitingMemory, kernelDim, i, j - 1);
        }
    }

    // printf("\nMult\n");
    // int cnt = 0;

    // Multiply the input to the weight and accumulate to the output
    for (int i= kernelDim * kernelDim - 1; i >= 0 ; i--){
        
        // printf("(%d * %d) + %d  \t|  ", inputMemory[i], weights[i], outputMemory[i]);
        // if(cnt == 3){
        //     printf("\n");
        //     cnt = 0;
        // } else {
        //     cnt ++;
        // }

        outputMemory[i + kernelDim] = int(inputMemory[i] * weights[i]) + outputMemory[i];
    }

    // Shift the input memory to the right
    for (int i = 0; i < kernelDim; i++) {
        for (int j= kernelDim - 1; j > 0; j--){
            inputMemory[i * kernelDim + j] = inputMemory[i * kernelDim + j - 1];
        }
    }

    // Shift the outWaitingMemory because of the skew in the output
    for (int j = 0; j < kernelDim; j++) {
        for (int i= kernelDim - 1; i > 0; i--){ // TODO: shift only the right-hand triangle
            mem2d(outWaitingMemory, kernelDim, i, j) = mem2d(outWaitingMemory, kernelDim, i - 1, j);
        }
    //    std::cout << std::hex << mem2d(outputMemory, kernelDim, kernelDim, j) << std::endl;
        mem2d(outWaitingMemory, kernelDim, 0, j) = (uint8_t)(mem2d(outputMemory, kernelDim, kernelDim, j) & 0xFF);
    }

    // printf("\n");
    // print_buf((int8_t*)outWaitingMemory, SA_W, SA_W);
    // printf("\n");

    // Return the output
    uint32_t result = 0;
    for (int i = 0; i < w_data; i++) {
        // printf("--> %d\n", mem2d(outWaitingMemory, kernelDim, kernelDim - i - 1, i ));
        result |=  mem2d(outWaitingMemory, kernelDim, kernelDim - i - 1, i ) << (8 * (w_data - i - 1));
//        std::cout << std::hex << (int) mem2d(outWaitingMemory, kernelDim, kernelDim - i - 1, i ) << ",";
    }
    return result;

}

}
