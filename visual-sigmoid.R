#!/usr/bin/env /c/Apps/Anaconda3/Scripts/Rscript
source("C://Home//tfR//f-sigmoid.R")
x = seq(-10,10,0.01)
cat(sprintf('%f %f\n',x, sigmoid(x)))
png("sigmoid.png", height = 800, width = 600)
plot(x,sigmoid(x))
dev.off()
system('explorer sigmoid.png')
