#!/usr/bin/env /c/Apps/Anaconda3/Scripts/Rscript

source("f-sigmoid.R")

w = 20; b = -10

x1 = 0; x2 = 0
answer = w*x1 + w*x2 + b
cat(sprintf('%d %d %3d %4.f\n',x1,x2,answer,sigmoid(answer)))

x1 = 1; x2 = 0
cat(sprintf('%d %d %3d %4.f\n',x1,x2,answer,sigmoid(answer)))

x1 = 0; x2 = 1
cat(sprintf('%d %d %3d %4.f\n',x1,x2,answer,sigmoid(answer)))

x1 = 1; x2 = 1
cat(sprintf('%d %d %3d %4.f\n',x1,x2,answer,sigmoid(answer)))
