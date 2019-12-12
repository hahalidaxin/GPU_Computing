#!/bin/bash
nvcc -arch sm_75 -o $1 $1.cu -run