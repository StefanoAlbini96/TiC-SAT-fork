//
// Created by alireza on 4/24/23.
//

#ifndef FVLLMONTITRANSFORMER_DEBUGGERFUNCTIONS_H
#define FVLLMONTITRANSFORMER_DEBUGGERFUNCTIONS_H
#include "util.h"
#include <iostream>
#include <fstream>
#include <cstdint>


void print_weight(uint32_t* kernel, int n_row, int n_col);
void blockWise2RowWise(const uint32_t * blockWise, uint32_t* rowWise, int n_row, int n_col);
void rowWise2BlockWise(const uint32_t* rowWise, uint32_t* blockWise, int n_row, int n_col);
void write_weight_to_file(const std::string& filename, uint32_t* kernel, int n_row, int n_col);
void read_weight_from_file(const std::string& filename, uint32_t* kernel, int n_row, int n_col);
void interleave_hidden_flag(uint32_t* kernel, int n_row, int n_col, uint32_t hidden_flag);
void interleave_hidden_flag_zero_free(uint32_t*& kernel, int n_row, int n_col, uint32_t hidden_flag);


void blockWise2Block3Dnano(const uint32_t * blockWise, uint32_t* rowWise, int n_row, int n_col);
// void blockWise2Block3Dnano_inputs(const uint32_t * blockWise, uint32_t* block3Dnano, int n_row, int n_col);
// void blockWise2Block3Dnano_inputs(const uint32_t *src, uint32_t *dst, int n_rows, int n_cols);
void blockWise2Block3Dnano_inputs(const uint32_t *src,
                                  uint32_t *dst,
                                  int rows,
                                  int cols);

#endif //FVLLMONTITRANSFORMER_DEBUGGERFUNCTIONS_H
