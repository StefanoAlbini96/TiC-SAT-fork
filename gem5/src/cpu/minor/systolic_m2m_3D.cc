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
// Constructor.


#include "cpu/minor/systolic_m2m_3D.hh"

#include <cstdint>
#include <iostream>

#include "arch/arm/system.hh"


namespace gem5
{


void print_buf_3Dnano(int8_t *ptr, int height, int width){
    int c = 0;
    for (int i = 0; i < height; i++) {
        for (int j = 0; j < width; j++) {
            printf("%d,  \t", ptr[c]);
            c++;
        
        }
        printf("\n");
    }
}



SystolicMatrixMultiplication_3D::SystolicMatrixMultiplication_3D(const SystolicMatrixMultiplication_3DParams &p)
    :   
      SimObject(p),
      kernelDim_h(p.kernel_dim_h),
      kernelDim_w(p.kernel_dim_w),
      n_3D_layers(p.n_3d_layers),
      weights(n_3D_layers, std::vector<int8_t>(kernelDim_h * kernelDim_w, 0)),
      inWaitingMemory(n_3D_layers, std::vector<int8_t>(kernelDim_h * kernelDim_h, 0)),
      inputMemory(n_3D_layers, std::vector<int8_t>(kernelDim_h * kernelDim_w, 0)),
      outWaitingMemory(n_3D_layers, std::vector<uint8_t>(kernelDim_w * kernelDim_w, 0)),
      outputMemory(n_3D_layers, std::vector<int32_t>(kernelDim_w * (kernelDim_h + 1), 0))
{
    // std::cout << "===========================" << std::endl;
    // std::cout << "Starting SMM 3D with params:" << std::endl;
    // std::cout << "SA_H        =   " << kernelDim_h << std::endl;
    // std::cout << "SA_W        =   " << kernelDim_w << std::endl;
    // std::cout << "3D layers   =   " << n_3D_layers << std::endl;
    // std::cout << "===========================" << std::endl;
}



bool SystolicMatrixMultiplication_3D::loadWeights(int idx, uint32_t val, int layer) {
    
    // printf("LOADING idx %d  |  layer = %d  |  val = %u\n", idx, layer, val);
    for (int i=0; i < w_data; i++){
        auto currVal = (int8_t)((val >> (8 * (w_data -i-1))) & 0xff);
        // printf("%d\n", currVal);
        weights[layer][idx + i] = currVal;
    }


    // print_buf_3Dnano((int8_t*)weights[1].data(), kernelDim_h, kernelDim_w);
    // printf("\n");

    if (val!=0)
        non_zero_tile = true;
    return non_zero_tile;
}


// Puts the new word of input and also reads a result word
uint32_t SystolicMatrixMultiplication_3D::inputQueue_3Dnano(int col, uint32_t val, int out_cnt) {


    // printf("\nEnqueueing in col = %d  | val = %u |  out_Cnt = %d\n", col, val, out_cnt);
    // print_buf(inWaitingMemory, kernelDim_h, kernelDim_h);
    // printf("\n");

    // std::cout << "IN = " << val << " --> ";
    // int8_t *res_qk = (int8_t*)&val; 
    // for(int j=0; j<4; j++){
    //     printf("%d, ", res_qk[j]);
    // }
    // printf("\n");


    for (int layer = 0; layer < n_3D_layers; layer++){
        // printf("\nLayer = %d\n", layer);

        // Split the input to an array
        // printf("New val --> ");
        for (int i=0; i < w_data; i++){
            auto currVal = (int8_t)((val >> (8 * (w_data - i -1))) & 0xff);
            // printf("%d, ", currVal);
            int row_index = (col*w_data+i);
            mem2d(inWaitingMemory[layer], kernelDim_h, row_index, kernelDim_h - row_index - 1) = currVal; // off-diagonal of the waiting memory
        }
        // printf("\n");

        // print_buf_3Dnano((int8_t*)inWaitingMemory[0].data(), kernelDim_h, kernelDim_h);
        // print_buf(inWaitingMemory, kernelDim_h, kernelDim_h);   
    }


    // printf("reading result for col = %d and cnt = %d\n", col, output_cnt);

    int col_cnt = out_cnt % max_col_out;
    int layer_cnt = out_cnt / max_col_out;
    // printf("Reading out_cnt = %d  | col_cnt = %d  | layer_cnt = %d\n", out_cnt, col_cnt, layer_cnt);

    // Return the output
    uint32_t result = 0;
    // int result_idx = ((col+1)%MAX_COL) * w_data;
    int result_idx = ((col_cnt+1)%max_col_out) * w_data;
    
    // int row_start = (1 - col_cnt) * w_data;
    int row_start = (max_col_out - 1 - col_cnt) * w_data;
    int col_start = kernelDim_w - 1 - row_start;


    // printf("Row start = (1 - %d) * 4 = %d\n", col_cnt, row_start);
    // printf("Col start =  %d - 1 - %d = %d\n", kernelDim_w, row_start, col_start);

    // printf("LAY 0: \n");
    // print_buf_3Dnano((int8_t*)outWaitingMemory[0], kernelDim_w, kernelDim_w);
    // printf("\nLAY 1: \n");
    // print_buf_3Dnano((int8_t*)outWaitingMemory[1], kernelDim_w, kernelDim_w);


    // printf("Ciao\n");
    // printf("result idx = ((%d + 1) %d) * %d = %d\n", col, max_col_out, w_data, ((col+1)%max_col_out) * w_data);
    for (int i = 0; i < w_data; i++) {


        // int row = row_start + i;
        // int col = col_start - i;

        int row = row_start + (w_data-1-i);;
        int col = col_start - (w_data-1-i);

        // printf("row = %d\n", row);
        // printf("col = %d\n", col);
        result |=  mem2d(outWaitingMemory[layer_cnt], kernelDim_w, row, col ) << (8 * (w_data - i - 1));
    }

    // printf("RESULT layer [%d], col [%d]\n", layer_cnt, col_start); 
    // int8_t *res = (int8_t*)&result; 
    // for(int j=0; j<w_data; j++){
    //     printf("%d, ", res[j]);
    // }
    // printf("\n");
    // printf("%u\n", result);

    return result;
}


void SystolicMatrixMultiplication_3D::printWeights() {
    std::cout << std::hex << (uint32_t) weights[0][0] << std::endl;
}




uint32_t SystolicMatrixMultiplication_3D::streamInOut_3Dnano(uint32_t val, int out_cnt) {


    // printf("\nINPUTTING val = %u  | out_Cnt = %d\n", val, out_cnt);

    for (int layer = 0; layer < n_3D_layers; layer++){
        // printf("---- LAYER %d\n", layer);

        non_zero_tile = false;
        int col = max_col - 1;
        // Split the input to an array
        // printf("New val --> ");
        for (int i=0; i < w_data; i++){
            auto currVal = (int8_t)((val >> (8 * (w_data - i -1))) & 0xff);
            // printf("%d, ", currVal);
            int row_index = (col*w_data+i);
            mem2d(inWaitingMemory[layer], kernelDim_h, row_index, kernelDim_h - row_index - 1) = currVal; // off-diagonal of the waiting memory
        }
        // printf("\n");

        // printf("\n----- \nWAIT\n");
        // print_buf(inWaitingMemory, kernelDim_h, kernelDim_h);

        // printf("\nINPUT\n");
        // print_buf(inputMemory, kernelDim_h, kernelDim_w);

        // Shift the waiting memory to the right for skewing
        for (int i = 0; i < kernelDim_h; i++) {
            mem2d(inputMemory[layer], kernelDim_w, i, 0) = mem2d(inWaitingMemory[layer], kernelDim_h, i, kernelDim_h - 1);
            for (int j= kernelDim_h - 1; j > 0; j--){ // TODO: shift only the right-hand triangle
                mem2d(inWaitingMemory[layer], kernelDim_h, i, j) = mem2d(inWaitingMemory[layer], kernelDim_h, i, j - 1);
            }
        }

        // printf("\n----- \nWAIT\n");
        // print_buf(inWaitingMemory, kernelDim_h, kernelDim_h);

        // printf("\nINPUT\n");
        // print_buf(inputMemory, kernelDim_h, kernelDim_w);



        // printf("\nMult\n");
        int cnt = 0;

        // printf("\n---------------------------------------------------------\n");
        // Multiply the input to the weight and accumulate to the output
        for (int i= kernelDim_h * kernelDim_w - 1; i >= 0 ; i--){

            // printf("(%d * %d) + %d  \t|  ", inputMemory[layer][i], weights[layer][i], outputMemory[layer][i]);
            // if(cnt == (kernelDim_w-1)){
            //     printf("\n");
            //     cnt = 0;
            // } else {
            //     cnt ++;
            // }

            outputMemory[layer][i + kernelDim_w] = int(inputMemory[layer][i] * weights[layer][i]) + outputMemory[layer][i];
        }

        // printf("\n----- \nWAIT\n");
        // print_buf(inWaitingMemory, kernelDim_h, kernelDim_h);

        // Shift the input memory to the right
        for (int i = 0; i < kernelDim_h; i++) {
            for (int j= kernelDim_w - 1; j > 0; j--){
                inputMemory[layer][i * kernelDim_w + j] = inputMemory[layer][i * kernelDim_w + j - 1];
            }
        }


        // printf("\n----- \nWAIT\n");
        // print_buf(inWaitingMemory, kernelDim_h, kernelDim_h);

        // printf("\nINPUT\n");
        // print_buf(inputMemory, kernelDim_h, kernelDim_w);


        // Shift the outWaitingMemory because of the skew in the output
        for (int j = 0; j < kernelDim_w; j++) {
            for (int i= kernelDim_w - 1; i > 0; i--){ // TODO: shift only the right-hand triangle
                mem2d(outWaitingMemory[layer], kernelDim_w, i, j) = mem2d(outWaitingMemory[layer], kernelDim_w, i - 1, j);
            }
        //    std::cout << std::hex << mem2d(outputMemory, KERNEL_DIM, KERNEL_DIM, j) << std::endl;
            mem2d(outWaitingMemory[layer], kernelDim_w, 0, j) = (uint8_t)(mem2d(outputMemory[layer], kernelDim_w, kernelDim_h, j) & 0xFF);
        }


        // printf("\n----- \nOUT WAIT\n");
        // print_buf((int8_t*)outWaitingMemory, kernelDim_w, kernelDim_w);



        // printf("\n----- \nOUT WAIT\n");
        // print_buf(outputMemory, kernelDim_w, kernelDim_h + 1);

        // printf("\n----- \nWAIT\n");
        // print_buf(inWaitingMemory, kernelDim_h, kernelDim_h);

    }

    // printf("\n");
    // print_buf_3Dnano((int8_t*)outWaitingMemory, kernelDim_w, kernelDim_w);
    // printf("\n");

    // printf("reading result for cnt = %d\n", output_cnt);
    int col_cnt = out_cnt % max_col_out;
    int layer_cnt = out_cnt / max_col_out;
    // printf("Reading out_cnt = %d  | col_cnt = %d  | layer_cnt = %d\n", out_cnt, col_cnt, layer_cnt);


    // printf("LAY 0: \n");
    // print_buf_3Dnano((int8_t*)outWaitingMemory[0].data(), kernelDim_w, kernelDim_w);
    // printf("\nLAY 1: \n");
    // print_buf_3Dnano((int8_t*)outWaitingMemory[1], kernelDim_w, kernelDim_w);

    // Return the output
    uint32_t result = 0;
    for (int i = 0; i < w_data; i++) {
        // printf("--> %d\n", mem2d(outWaitingMemory, KERNEL_DIM, KERNEL_DIM - i - 1, i ));
        result |=  mem2d(outWaitingMemory[layer_cnt], kernelDim_w, kernelDim_w - i - 1, i ) << (8 * (w_data - i - 1));
//        std::cout << std::hex << (int) mem2d(outWaitingMemory, KERNEL_DIM, KERNEL_DIM - i - 1, i ) << ",";
    }

    // printf("RESULT layer [%d]\n", layer_cnt);
    // int8_t *res = (int8_t*)&result; 
    // for(int j=0; j<w_data; j++){
    //     printf("%d, ", res[j]);
    // }
    // printf("\n");
    // printf("%u\n", result);

    
    return result;

}

}



