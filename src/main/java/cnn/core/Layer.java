package cnn.core;

import cnn.utils.MathUtils;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.Serializable;

public class Layer implements Serializable {

	private static final long serialVersionUID = -5747622503947497069L;

	private static Logger logger = LoggerFactory.getLogger(Layer.class);

	private LayerType type;//层次类型
	private int outMapNum;//图的输出数量
	private Size mapSize;//图的大小
	private Size kernelSize;//内卷核大小
	private Size scaleSize;//池化大小
	private double[][][][] kernel;//对应的内卷核参数
	private double[] bias;//偏置值

	private double[][][][] outmaps;

	private double[][][][] errors;

	private static int recordInBatch = 0;

	private int classNum = -1;//分类数量

	private Layer() {
		//
	}

	public static void prepareForNewBatch() {
		recordInBatch = 0;
	}

	public static void prepareForNewRecord() {
		recordInBatch++;
	}
	//创建输入层
	public static Layer buildInputLayer(Size mapSize) {
		Layer layer = new Layer();
		layer.type = LayerType.input;
		layer.outMapNum = 1;
		layer.setMapSize(mapSize);
		return layer;
	}
	//创建 卷化层（过滤器数量，大小）
	public static Layer buildConvLayer(int outMapNum, Size kernelSize) {
		Layer layer = new Layer();
		layer.type = LayerType.conv;
		layer.outMapNum = outMapNum;
		layer.kernelSize = kernelSize;
		return layer;
	}
	//创建池化层（大小）
	public static Layer buildSampLayer(Size scaleSize) {
		Layer layer = new Layer();
		layer.type = LayerType.samp;
		layer.scaleSize = scaleSize;
		return layer;
	}
	//创建全连接层
	public static Layer buildOutputLayer(int classNum) {
		Layer layer = new Layer();
		layer.classNum = classNum;
		layer.type = LayerType.output;
		layer.mapSize = new Size(1, 1);
		layer.outMapNum = classNum;
		//		int outMapNum = 1;
		//		while ((1 << outMapNum) < classNum)
		//			outMapNum += 1;
		//		layer.outMapNum = outMapNum;
		logger.info("outMapNum:{}", layer.outMapNum);
		return layer;
	}

	public Size getMapSize() {
		return mapSize;
	}

	public void setMapSize(Size mapSize) {
		this.mapSize = mapSize;
	}

	public LayerType getType() {
		return type;
	}

	public int getOutMapNum() {
		return outMapNum;
	}

	public void setOutMapNum(int outMapNum) {
		this.outMapNum = outMapNum;
	}

	public Size getKernelSize() {
		return kernelSize;
	}

	public Size getScaleSize() {
		return scaleSize;
	}

	enum LayerType {
		//输入，输出，卷积，池化
		input, output, conv, samp
	}

	public static class Size implements Serializable {

		private static final long serialVersionUID = -209157832162004118L;

		public final int x;
		public final int y;

		public Size(int x, int y) {
			this.x = x;
			this.y = y;
		}

		@Override
		public String toString() {
			StringBuilder s = new StringBuilder("Size(").append(" x = ").append(x).append(" y= ").append(y).append(")");
			return s.toString();
		}
		//池化操作（对大小）
		public Size divide(Size scaleSize) {
			int x = this.x / scaleSize.x;
			int y = this.y / scaleSize.y;
			if (x * scaleSize.x != this.x || y * scaleSize.y != this.y)
				throw new RuntimeException(this + "scalesize:" + scaleSize);
			return new Size(x, y);
		}
	//输出卷化后大小
		public Size subtract(Size size, int append) {
			int x = this.x - size.x + append;
			int y = this.y - size.y + append;
			return new Size(x, y);
		}

	}
	//初始化卷化核
	public void initKernel(int frontMapNum) {
		//		int fan_out = getOutMapNum() * kernelSize.x * kernelSize.y;
		//		int fan_in = frontMapNum * kernelSize.x * kernelSize.y;
		//		double factor = 2 * Math.sqrt(6 / (fan_in + fan_out));
		this.kernel = new double[frontMapNum][outMapNum][kernelSize.x][kernelSize.y];
		for (int i = 0; i < frontMapNum; i++)
			for (int j = 0; j < outMapNum; j++)
				kernel[i][j] = MathUtils.randomMatrix(kernelSize.x, kernelSize.y, true);
	}
	//初始化全连接的卷化核矩阵
	public void initOutputKerkel(int frontMapNum, Size size) {
		kernelSize = size;
		//		int fan_out = getOutMapNum() * kernelSize.x * kernelSize.y;
		//		int fan_in = frontMapNum * kernelSize.x * kernelSize.y;
		//		double factor = 2 * Math.sqrt(6 / (fan_in + fan_out));
		this.kernel = new double[frontMapNum][outMapNum][kernelSize.x][kernelSize.y];
		for (int i = 0; i < frontMapNum; i++)
			for (int j = 0; j < outMapNum; j++)
				kernel[i][j] = MathUtils.randomMatrix(kernelSize.x, kernelSize.y, false);
	}
	//初始化偏置值（【】）
	public void initBias(int frontMapNum) {
		this.bias = MathUtils.randomArray(outMapNum);
	}

	public void initOutmaps(int batchSize) {
		outmaps = new double[batchSize][outMapNum][mapSize.x][mapSize.y];
	}

	public void setMapValue(int mapNo, int mapX, int mapY, double value) {
		outmaps[recordInBatch][mapNo][mapX][mapY] = value;
	}

	static int count = 0;
	//设置图的值
	public void setMapValue(int mapNo, double[][] outMatrix) {
		// Log.i(type.toString());
		// Util.printMatrix(outMatrix);
		outmaps[recordInBatch][mapNo] = outMatrix;
	}
	//获取图
	public double[][] getMap(int index) {
		return outmaps[recordInBatch][index];
	}

	public double[][] getKernel(int i, int j) {
		return kernel[i][j];
	}

	public void setError(int mapNo, int mapX, int mapY, double value) {
		errors[recordInBatch][mapNo][mapX][mapY] = value;
	}

	public void setError(int mapNo, double[][] matrix) {
		// Log.i(type.toString());
		// Util.printMatrix(matrix);
		errors[recordInBatch][mapNo] = matrix;
	}

	public double[][] getError(int mapNo) {
		return errors[recordInBatch][mapNo];
	}

	public double[][][][] getErrors() {
		return errors;
	}

	public void initErros(int batchSize) {
		errors = new double[batchSize][outMapNum][mapSize.x][mapSize.y];
	}

	public void setKernel(int lastMapNo, int mapNo, double[][] kernel) {
		this.kernel[lastMapNo][mapNo] = kernel;
	}

	public double getBias(int mapNo) {
		return bias[mapNo];
	}

	public void setBias(int mapNo, double value) {
		bias[mapNo] = value;
	}

	public double[][][][] getMaps() {
		return outmaps;
	}

	public double[][] getError(int recordId, int mapNo) {
		return errors[recordId][mapNo];
	}

	public double[][] getMap(int recordId, int mapNo) {
		return outmaps[recordId][mapNo];
	}

	public int getClassNum() {
		return classNum;
	}

	public double[][][][] getKernel() {
		return kernel;
	}

}
