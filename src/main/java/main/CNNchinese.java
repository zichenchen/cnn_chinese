package main;

import cnn.core.CNN;
import cnn.core.Layer;
import cnn.data.DataSet;
import cnn.utils.ConcurentRunner;

import java.util.Date;

public class CNNchinese {
    private static final String MODEL_NAME = "chinese/model/model.cnn";

    private static final String TRAIN_DATA = "chinese/train2.data";

    private static final String TEST_DATA = "chinese/test.data";

    private static final String TEST_PREDICT = "chinese/predict.data";
    public static void main(String[] args) {
        System.out.println("训练阶段：");
      //  runTrain();
       System.err.println("测试阶段：");
       runTest();
        ConcurentRunner.stop();
    }
    public static void runTrain() {
        // 构建网络层次结构
        CNN.LayerBuilder builder = new CNN.LayerBuilder();
        builder.addLayer(Layer.buildInputLayer(new Layer.Size(28, 28))); // 输入层输出map大小为28×28
        builder.addLayer(Layer.buildConvLayer(6, new Layer.Size(5, 5))); // 卷积层输出map大小为24×24,24=28+1-5
        builder.addLayer(Layer.buildSampLayer(new Layer.Size(2, 2))); // 采样层输出map大小为12×12,12=24/2
        builder.addLayer(Layer.buildConvLayer(12, new Layer.Size(5, 5))); // 卷积层输出map大小为8×8,8=12+1-5
         builder.addLayer(Layer.buildSampLayer(new Layer.Size(2, 2))); // 采样层输出map大小为4×4,4=8/2
        builder.addLayer(Layer.buildOutputLayer(50));
        CNN cnn = new CNN(builder, 50);
        // 加载训练数据
        DataSet dataset = DataSet.loadchinese(TRAIN_DATA, "，", 784);
        // 开始训练模型&
        long t1= new Date().getTime();
        cnn.train(dataset, 5);
        long t2= new Date().getTime();
        System.out.println(t2-t1);
        // 保存训练好的模型
       cnn.saveModel(MODEL_NAME);
        dataset.clear();
    }
    public static void runTest() {
        // 加载训练好的模型
        CNN cnn = CNN.loadModel(MODEL_NAME);
        // 加载测试数据
        DataSet testSet = DataSet.load(TEST_DATA, "，", -1);
        // 预测结果
        cnn.predict(testSet, TEST_PREDICT);
        testSet.clear();
    }
}
