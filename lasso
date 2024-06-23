# 安装并加载必要的包
# 如果没有安装这些包，请取消注释以下安装命令
# install.packages("glmnet")
# install.packages("iml")
# install.packages("shapviz")

library(glmnet)
library(iml)
library(shapviz)

# 读取并准备数据
tpms_transposed <- read.table("MachineLearning.txt", sep = "\t", row.names = 1, check.names = FALSE, stringsAsFactors = FALSE, header = TRUE)
tpms_transposed <- tpms_transposed[, c("CDKN1C", "ADRA1A", "BCL2A1", "DPP4", "CACNA1B", "TFPI", "KCNA5", "CYB5A", "Label")]

# 读取数据
my_data <- tpms_transposed

# 查看数据结构
str(my_data)

# 确保最后一列是Label
if (!("Label" %in% colnames(my_data))) {
  stop("数据框中没有 'Label' 列，请检查数据。")
}

# 标准化特征数据（去除Label列）
features <- my_data[, -ncol(my_data)]
features <- scale(features)

# 将标准化后的数据和Label列组合成新的数据框
my_data_scaled <- data.frame(features, Label = my_data$Label)

# 划分训练集和测试集
set.seed(123)
inTrain <- createDataPartition(y = my_data_scaled$Label, p = 0.6, list = FALSE)
traindata <- my_data_scaled[inTrain, ]
testdata <- my_data_scaled[-inTrain, ]

# 特征名称
feature_names <- colnames(traindata)[-ncol(traindata)]

# 准备数据
x <- as.matrix(traindata[, feature_names])
y <- traindata$Label

# 使用交叉验证选择最佳lambda
cv_model <- cv.glmnet(x, y, alpha = 0.5, family = "binomial")

# 打印最佳lambda值
best_lambda <- cv_model$lambda.min
print(paste("Best lambda:", best_lambda))

# 训练GLMNET模型
model_glmnet <- glmnet(x, y, alpha = 0.5, family = "binomial", lambda = best_lambda)

# 生成训练集预测值
traindata$pred_glmnet <- as.numeric(predict(model_glmnet, newx = x, type = "response"))

# 计算训练集AUC
ROC_train_glmnet <- round(auc(response = traindata$Label, predictor = traindata$pred_glmnet), 4)
print(paste("Train AUC (GLMNET):", ROC_train_glmnet))

# 在测试集上生成预测值
x_test <- as.matrix(testdata[, feature_names])
testdata$pred_glmnet <- as.numeric(predict(model_glmnet, newx = x_test, type = "response"))

# 计算测试集AUC
ROC_test_glmnet <- round(auc(response = testdata$Label, predictor = testdata$pred_glmnet), 4)
print(paste("Test AUC (GLMNET):", ROC_test_glmnet))

# 创建混淆矩阵并计算性能指标
predicted_classes_glmnet <- ifelse(testdata$pred_glmnet > 0.5, 1, 0)
confusion_matrix_glmnet <- confusionMatrix(as.factor(predicted_classes_glmnet), as.factor(testdata$Label))
print(confusion_matrix_glmnet)

# 提取准确率、召回率、特异性和F1分数
accuracy_glmnet <- confusion_matrix_glmnet$overall["Accuracy"]
recall_glmnet <- confusion_matrix_glmnet$byClass["Recall"]
specificity_glmnet <- confusion_matrix_glmnet$byClass["Specificity"]
f1_glmnet <- confusion_matrix_glmnet$byClass["F1"]

cat("准确率 (GLMNET):", accuracy_glmnet, "\n")
cat("召回率 (GLMNET):", recall_glmnet, "\n")
cat("特异性 (GLMNET):", specificity_glmnet, "\n")
cat("F1分数 (GLMNET):", f1_glmnet, "\n")
