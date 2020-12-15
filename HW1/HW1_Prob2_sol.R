library(ggplot2)

path = "D:\\ISU\\COMS 574 - Introduction to Machine Learning\\HW\\HW1\\"

dt = read.csv(paste(path, "housingdata.csv", sep = ""), header = T)
varname = names(dt)

# 2. (a)

for (i in c(1:13)){
 print(ggplot(dt, aes_string(x=varname[i], y=varname[14])) + 
         ylim(0,max(dt$MEDV+2))+
         ggtitle(varname[i]) +
         geom_point())
}


# 2 (b)

dt_train = dt[1:400,]
dt_test = dt[401:dim(dt)[1],]
varn = c("AGE", "INDUS", "NOX", "RM", "TAX")

reg_res = list()
tr_mse = list()
k = 1
for (i in c(0:length(varn))){
  comb = combn(varn, i)
  for (j in c(1:dim(comb)[2])){
    if (dim(comb)[1]==0){
      formu = "MEDV ~ 1"
      comb = matrix(c("Intercept"), nrow = 1, ncol = 1)
    } else{
      formu = paste("MEDV ~ 1", paste((comb[,j]), collapse = "+"), sep = "+")  
    }
    
    reg_res[[k]] = lm(as.formula(formu), data = dt_train)
    tr_mse[[k]] = list(i = i, var = paste((comb[,j]), collapse = ","), 
                       tr_mse = mean(reg_res[[k]]$residuals^2), 
                       vel_mse = mean((predict(reg_res[[k]], newdata = dt_test)-dt_test$MEDV)^2))
    k = k+1
  }
  
}

res = do.call(rbind.data.frame, tr_mse)
names(res) = c("subset", "variables", "tr_mse", "ts_mse")
trset = NULL
for (i in c(0:length(varn))){
  aa = which(res$tr_mse == res[res$subset == i,][which.min(res[res$subset == i,3]),3])
  trset = rbind(trset, res[aa,])
  res1 = reg_res[[aa]]
  len1 = length(res1$coefficients)
  if (i==0){
    print(paste("For subset model ", i, ":   y^hat = ", round(res1$coefficients[1], 3) ,
                "   with MSE = ", round(res[aa,3],3))) 
  } else {
    
    print(paste("For subset model ", i, ":   y^hat = ", round(res1$coefficients[1], 3), " + " ,
              paste(round(res1$coefficients[2:len1], 3), 
                    names(res1$coefficients)[2:len1], sep = " * ", collapse = " + "),
      "   with training MSE = ", round(res[aa, 3],3)))
  }
}

print(trset)

#2 (c)
valset = NULL
for (i in c(0:length(varn))){
  aa = which(res$ts_mse == res[res$subset == i,][which.min(res[res$subset == i,4]),4])
  valset = rbind(valset, res[aa,])
  res1 = reg_res[[aa]]
  len1 = length(res1$coefficients)
  if (i==0){
    print(paste("For subset model ", i, ":   y^hat = ", round(res1$coefficients[1], 3) ,
                "   with MSE = ", round(res[aa,4],3))) 
  } else {
    
    print(paste("For subset model ", i, ":   y^hat = ", round(res1$coefficients[1], 3), " + " ,
                paste(round(res1$coefficients[2:len1], 3), 
                      names(res1$coefficients)[2:len1], sep = " * ", collapse = " + "),
                "   with validation MSE = ", round(res[aa,4],3)))
  }
}

print(valset)


# 2 (c) (ii)
plot(trset$subset, trset$tr_mse, type = "l", xlab = "i", lwd = 2, col = 2,
     ylab = "Training Loss (MSE)")
title("Training loss - best subsets")


plot(valset$subset, valset$tr_mse, type = "l", xlab = "i", lwd = 2, col = 2,
     ylab = "Validation Loss (MSE)")
title("Validation loss - best subsets")

###############

# 2 (c) (iii)

reg_res_all = list()
tr_mse_all = list()
k = 1
for (i in c(0:length(varn))){
  comb = combn(varn, i)
  for (j in c(1:dim(comb)[2])){
    if (dim(comb)[1]==0){
      formu = "MEDV ~ 1"
      comb = matrix(c("Intercept"), nrow = 1, ncol = 1)
    } else{
      formu = paste("MEDV ~ 1", paste((comb[,j]), collapse = "+"), sep = "+")  
    }
    
    reg_res_all[[k]] = lm(as.formula(formu), data = dt)
    trmse = mean(reg_res_all[[k]]$residuals^2)
    tr_mse_all[[k]] = list(i = i, var = paste((comb[,j]), collapse = ","), 
                       tr_mse_all = trmse)
    k = k+1
  }
  
}

res_all = do.call(rbind.data.frame, tr_mse_all)
names(res_all) = c("subset", "variables", "tr_mse")
res_all$cp = res_all$tr_mse + 2*res_all$subset*res_all$tr_mse[nrow(res_all)] / nrow(dt)


trset_cp = NULL
for (i in c(0:length(varn))){
  aa = which(res_all$cp == res_all[res_all$subset == i,][which.min(res_all[res_all$subset == i,4]),4])
  trset_cp = rbind(trset_cp, res_all[aa,])
  res_all1 = reg_res_all[[aa]]
  len1 = length(res_all1$coefficients)
  if (i==0){
    print(paste("For subset model ", i, ":   y^hat = ", round(res_all1$coefficients[1], 3) ,
                "   with MSE = ", round(res_all[aa,3],3))) 
  } else {
    
    print(paste("For subset model ", i, ":   y^hat = ", round(res_all1$coefficients[1], 3), " + " ,
                paste(round(res_all1$coefficients[2:len1], 3), 
                      names(res_all1$coefficients)[2:len1], sep = " * ", collapse = " + "),
                "   with Mallow's Cp = ", round(res_all[aa, 4],3)))
  }
}

plot(trset_cp$subset, trset_cp$cp, type = "l", xlab = "i", lwd = 2, col = 2,
     ylab = "Mallow's Cp")
title("Mallow's Cp - best subsets")

#####
# 2 (c) (iv)

library(glmnet)
dt_train = dt[1:400,]
dt_test = dt[401:dim(dt)[1],]
varn = c("AGE", "INDUS", "NOX", "RM", "TAX")

lambda = 0
l2_res = NULL

loop = TRUE

while (loop) {
  a = glmnet(x = as.matrix(dt_train[,varn]), y = dt_train[,"MEDV"], 
             alpha = 0, lambda = lambda)
  pred = predict(a, as.matrix(dt_test[,varn]))
  l2 = mean((dt_test$MEDV - pred)^2) + lambda * sum(a$beta^2)
  l2_res = rbind(l2_res, c(lambda, l2))
  lambda = lambda + 10
  if(sum(a$beta^2) < 1e-5) loop = FALSE
}

l2_res = as.data.frame(l2_res)
plot(l2_res$V1, l2_res$V2, type = "l", xlab = "Lambda", lwd = 2, col = 2,
     ylab = "L2 total complexity")
title("L2 Panalty")



lambda = 0
l1_res = NULL

loop = TRUE

while (loop) {
  a = glmnet(x = as.matrix(dt_train[,varn]), y = dt_train[,"MEDV"], 
             alpha = 1, lambda = lambda)
  pred = predict(a, as.matrix(dt_test[,varn]))
  l1 = mean((dt_test$MEDV - pred)^2) + lambda * sum(abs(a$beta))
  l1_res = rbind(l1_res, c(lambda, l1))
  lambda = lambda + 0.02
  if(sum(abs(a$beta)) < 1e-8) loop = FALSE
}

l1_res = as.data.frame(l1_res)
plot(l1_res$V1, l1_res$V2, type = "l", xlab = "Lambda", lwd = 2, col = 2,
     ylab = "L1 total complexity")
title("L1 Panalty")



#### # 2 (d)

dt_train = dt[1:400,]
dt_test = dt[401:dim(dt)[1],]
varn = c("CRIM",    "ZN",      "INDUS",   "CHAS",    "NOX",     "RM",
         "AGE",     "DIS",     "RAD",     "TAX",     "PTRATIO", "B" ,      "LSTAT")


varinclude = c()
varexlude = varn
fi_res = list()
fi_reg = list()
k=2

formu = "MEDV ~ 1"
tem_reg = lm(as.formula(formu), data = dt_train)
temp_res = list(0, "intercept", 
                     mean(tem_reg$residuals^2), 
                     mean((predict(tem_reg, newdata = dt_test)-dt_test$MEDV)^2),
                     formu)


fi_reg[[1]] = tem_reg
fi_res[[1]] = c(temp_res)


while (!is.null(varexlude) & length(varexlude)>0) {
  tem_reg = list()
  temp_res = list()
 
  
  for (i in c(1:length(varexlude))){
    
    if (is.null(varinclude)){
      formu = paste("MEDV ~ 1", varexlude[i], sep = "+")  
    } else {
      formu = paste("MEDV ~ 1", paste(varinclude, collapse = "+"), varexlude[i], sep = "+")    
    }
  
    tem_reg[[i]] = lm(as.formula(formu), data = dt_train)
    temp_res[[i]] = list(length(varinclude)+1, varexlude[i], 
                 mean(tem_reg[[i]]$residuals^2), 
                 mean((predict(tem_reg[[i]], newdata = dt_test)-dt_test$MEDV)^2),
                 formu)
    
  }
  
  temp_res = do.call(rbind.data.frame, temp_res)
  names(temp_res) = c("nvar", "var_include", "tr_mse", "ts_mse", "formula")
  temp_res$var_include = as.character(temp_res$var_include)
  temp_res$formula = as.character(temp_res$formula)
  varinclude = c(varinclude, as.character(temp_res[which.min(temp_res$ts_mse), 2]))
  varexlude = varexlude[!(varexlude %in% varinclude)]
  
  fi_reg[[k]] = tem_reg[[which.min(temp_res$ts_mse)]]
  fi_res[[k]] = c(temp_res[which.min(temp_res$ts_mse),])

  k=k+1
}    
    
    
    
    
    
fi_res = do.call(rbind.data.frame, fi_res)
names(fi_res) = c("nvar", "var_include", "tr_mse", "ts_mse", "formula")




plot(fi_res$nvar, fi_res$tr_mse, type = "l", xlab = "Number of features", lwd = 2, col = 2,
     ylab = "Training MSE")
title("Training MSE for all subsets")


plot(fi_res$nvar, fi_res$ts_mse, type = "l", xlab = "Number of features", lwd = 2, col = 2,
     ylab = "Test MSE")
title("Test MSE for all subsets")

print(fi_res)

print(fi_res[which.min(fi_res$ts_mse),])

aa = which.min(fi_res$ts_mse)
res_131 = fi_reg[[aa]]
len1 = length(res_131$coefficients)
print(paste("For subset model ", fi_res[aa,]$formula, ":   y^hat = ", round(res_131$coefficients[1], 3), " + " ,
            paste(round(res_131$coefficients[2:len1], 3), 
                  names(res_131$coefficients)[2:len1], sep = " * ", collapse = " + "),
            "   with training MSE = ", round(fi_res[aa,3],3), 
            " and validation MSE = ", round(fi_res[aa,4],3)))




######

###2 (d) (ii)


dt_train = dt[1:400,]
dt_test = dt[401:dim(dt)[1],]
varn = c("CRIM",    "ZN",      "INDUS",   "CHAS",    "NOX",     "RM",
         "AGE",     "DIS",     "RAD",     "TAX",     "PTRATIO", "B" ,      "LSTAT")


varinclude = varn
varexlude = c()
fi_res = list()
fi_reg = list()
k=2

formu = paste("MEDV ~ 1", paste(varinclude, collapse = "+"), sep = "+")    
tem_reg = lm(as.formula(formu), data = dt_train)
temp_res = list(13, "None", 
                mean(tem_reg$residuals^2), 
                mean((predict(tem_reg, newdata = dt_test)-dt_test$MEDV)^2),
                formu)


fi_reg[[1]] = tem_reg
fi_res[[1]] = c(temp_res)


while (!is.null(varinclude) & length(varinclude)>0) {
  tem_reg = list()
  temp_res = list()
  
  
  for (i in c(1:length(varinclude))){

    if (is.null(varinclude) | length(varinclude)==1){
      formu = "MEDV ~ 1"  
    } else {
      formu = paste("MEDV ~ 1", paste(varinclude[-i], collapse = "+"), sep = "+")    
    }
    
    tem_reg[[i]] = lm(as.formula(formu), data = dt_train)
    temp_res[[i]] = list(length(varinclude)-1, varinclude[i], 
                         mean(tem_reg[[i]]$residuals^2), 
                         mean((predict(tem_reg[[i]], newdata = dt_test)-dt_test$MEDV)^2),
                         formu)
    
  }
  
  temp_res = do.call(rbind.data.frame, temp_res)
  names(temp_res) = c("nvar", "var_exclude", "tr_mse", "ts_mse", "formula")
  temp_res$var_exclude = as.character(temp_res$var_exclude)
  temp_res$formula = as.character(temp_res$formula)
  varexlude = c(varexlude, as.character(temp_res[which.min(temp_res$ts_mse), 2]))
  varinclude = varinclude[!(varinclude %in% varexlude)]
  
  fi_reg[[k]] = tem_reg[[which.min(temp_res$ts_mse)]]
  fi_res[[k]] = c(temp_res[which.min(temp_res$ts_mse),])
  
  k=k+1
}    





fi_res = do.call(rbind.data.frame, fi_res)
names(fi_res) = c("nvar", "var_include", "tr_mse", "ts_mse", "formula")




plot(fi_res$nvar, fi_res$tr_mse, type = "l", xlab = "Number of features", lwd = 2, col = 2,
     ylab = "Training MSE")
title("Training MSE for all subsets")


plot(fi_res$nvar, fi_res$ts_mse, type = "l", xlab = "Number of features", lwd = 2, col = 2,
     ylab = "Test MSE")
title("Test MSE for all subsets")

print(fi_res)

print(fi_res[which.min(fi_res$ts_mse),])

aa = which.min(fi_res$ts_mse)
res_131 = fi_reg[[aa]]
len1 = length(res_131$coefficients)
print(paste("For subset model ", fi_res[aa,]$formula, ":   y^hat = ", round(res_131$coefficients[1], 3), " + " ,
            paste(round(res_131$coefficients[2:len1], 3), 
                  names(res_131$coefficients)[2:len1], sep = " * ", collapse = " + "),
            "   with training MSE = ", round(fi_res[aa,3],3), 
            " and validation MSE = ", round(fi_res[aa,4],3)))
