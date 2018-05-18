#!/usr/bin/env /c/Apps/Anaconda3/Scripts/Rscript
# XOR problem 
# modified
# Author : Yibeck Lee(yibeck.lee@gmail.com)
options(encoding = "utf-8")
x1 = c(0,0,1,1)
x2 = c(0,1,0,1)
y = c(0,1,1,0)
x = data.frame(cbind(x1,x2))
print(x)

# logistic model
model = glm(
	 formula=y ~ x1+x2
	,family='binomial' 
	)

summary(model)
pred = predict(model,data.frame(x1=0, x2=0), type="response")
print(paste('0 0 =',pred))
pred = predict(model,data.frame(x1=0, x2=1), type="response")
print(paste('0 1 =',pred))
pred = predict(model,data.frame(x1=1, x2=0), type="response")
print(paste('1 0 =',pred))
pred = predict(model,data.frame(x1=1, x2=1), type="response")
print(paste('1 1 =',pred))
# # ¸모델 평가
pred = predict(model, newdata=x, type="response")
pred
print("Stat Summary")
install.packages("ROCR")


install.packages("ROCR")
stat_summary <- prediction(pred, y)
stat_summary
library(ROCR)
png(file="C:/Home/tfR/roc-xor-logistic.png")
plot(performance(stat_summary, "tpr", "fpr"), col = "black", lty = 3, lwd = 3)
dev.off()
