package cnn.core;

import cnn.data.DataSet;
import cnn.utils.ConcurentRunner;
import cnn.utils.MathUtils;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.*;
import java.util.ArrayList;
import java.util.Iterator;
import java.util.List;
import java.util.concurrent.atomic.AtomicBoolean;

public class CNN implements Serializable {

	private static final long serialVersionUID = 337920299147929932L;
	//获取日志对象
	private static Logger logger = LoggerFactory.getLogger(CNN.class);

	private static double ALPHA = 0.01;
	protected static final double LAMBDA = 0;
	private List<Layer> layers;//层集合
	private int layerNum;//层数目

	private int batchSize;//
	private MathUtils.Operator divide_batchSize;

	private MathUtils.Operator multiply_alpha;

	private MathUtils.Operator multiply_lambda;
	//对CNN进行初始化
	public CNN(LayerBuilder layerBuilder, final int batchSize) {
		layers = layerBuilder.mLayers;
		layerNum = layers.size();
		this.batchSize = batchSize;
		setup(batchSize);
		initPerator();
	}
//功能函数
	private void initPerator() {

		divide_batchSize = new MathUtils.Operator() {

			private static final long serialVersionUID = 7424011281732651055L;

			@Override
			public double process(double value) {
				return value / batchSize;
			}

		};

		multiply_alpha = new MathUtils.Operator() {

			private static final long serialVersionUID = 5761368499808006552L;

			@Override
			public double process(double value) {

				return value * ALPHA;
			}

		};

		multiply_lambda = new MathUtils.Operator() {

			private static final long serialVersionUID = 4499087728362870577L;

			@Override
			public double process(double value) {

				return value * (1 - LAMBDA * ALPHA);
			}

		};

	}

	//参数（数据集合，重复次数）
	public void train(DataSet trainset, int repeat) {

		new Lisenter().start();

		for (int t = 0; t < repeat && !stopTrain.get(); t++) {
			//数据分批
			int epochsNum = trainset.size() / batchSize;
			if (trainset.size() % batchSize != 0)
				epochsNum++;
			int right = 0;
			int count = 0;
			logger.info("第{}次迭代，epochsNum: {}", t, epochsNum);

			for (int i = 0; i < epochsNum; i++) {
				int[] randPerm = MathUtils.randomPerm(trainset.size(), batchSize);
				Layer.prepareForNewBatch();
				//任意取10条数据进行学习
				for (int index : randPerm) {

					boolean isRight = train(trainset.getRecord(index));
					if (isRight)
						right++;
					count++;
					Layer.prepareForNewRecord();
				}
				double p1 = 1.0 * right / count;
				System.out.println((i+t*epochsNum)+" "+right+" "+count+" "+p1);
				updateParas();
			}
			double p = 1.0 * right / count;
			if (t % 10 == 1 && p > 0.96) {
				ALPHA = ALPHA * 0.5;
				logger.info("设置 alpha = {}", ALPHA);
			}

			logger.info("计算精度： {}/{}={}.", right, count, p);
		}

	}
	private static AtomicBoolean stopTrain;

	static class Lisenter extends Thread {

		Lisenter() {
			setDaemon(true);
			stopTrain = new AtomicBoolean(false);
		}

		@Override
		public void run() {
			logger.info("输入&符号停止训练.");
			while (true) {
				try {
					int a = System.in.read();
					if (a == '&') {
						stopTrain.compareAndSet(false, true);
						break;
					}
				} catch (IOException e) {
					e.printStackTrace();
				}
			}
			System.out.println("监听停止.");
		}

	}

	@SuppressWarnings("unused")
	private double test(DataSet trainset) {
		Layer.prepareForNewBatch();
		Iterator<DataSet.Record> iter = trainset.iter();
		int right = 0;
		while (iter.hasNext()) {
			DataSet.Record record = iter.next();
			forward(record);
			Layer outputLayer = layers.get(layerNum - 1);
			int mapNum = outputLayer.getOutMapNum();
			double[] out = new double[mapNum];
			for (int m = 0; m < mapNum; m++) {
				double[][] outmap = outputLayer.getMap(m);
				out[m] = outmap[0][0];
			}
			if (record.getLabel().intValue() == MathUtils.getMaxIndex(out))
				right++;
		}
		double p = 1.0 * right / trainset.size();
		logger.info("计算精度为：\t{}", p + "");
		return p;
	}

	public void predict(DataSet testset, String fileName) {
		logger.info("开始预测 ...");
		try {
			//			int max = layers.get(layerNum - 1).getClassNum();
			PrintWriter writer = new PrintWriter(new File(fileName));
			Layer.prepareForNewBatch();
			Iterator<DataSet.Record> iter = testset.iter();
			while (iter.hasNext()) {
				DataSet.Record record = iter.next();
				forward(record);
				Layer outputLayer = layers.get(layerNum - 1);

				int mapNum = outputLayer.getOutMapNum();
				double[] out = new double[mapNum];
				for (int m = 0; m < mapNum; m++) {
					double[][] outmap = outputLayer.getMap(m);
					out[m] = outmap[0][0];
				}
				//				int lable = MathUtils.binaryArray2int(out);
				int lable = MathUtils.getMaxIndex(out);
				//				if (lable >= max)
				//					lable = lable - (1 << (out.length - 1));
				writer.write(lable + "\n");
			}
			writer.flush();
			writer.close();
		} catch (IOException e) {
			throw new RuntimeException(e);
		}
		logger.info("完成预测 ...");
	}

	@SuppressWarnings("unused")
	private boolean isSame(double[] output, double[] target) {
		boolean r = true;
		for (int i = 0; i < output.length; i++)
			if (Math.abs(output[i] - target[i]) > 0.5) {
				r = false;
				break;
			}

		return r;
	}
//
	private boolean train(DataSet.Record record) {
		//进行数据设置
		forward(record);
		//反向传播设置
		boolean result = backPropagation(record);
		return result;
		// System.exit(0);
	}
	//反向传播
	private boolean backPropagation(DataSet.Record record) {
		boolean result = setOutLayerErrors(record);
		setHiddenLayerErrors();
		return result;
	}
//更新权值参数
	private void updateParas() {
		for (int l = 1; l < layerNum; l++) {
			Layer layer = layers.get(l);
			Layer lastLayer = layers.get(l - 1);
			switch (layer.getType()) {
			case conv:
			case output:
				updateKernels(layer, lastLayer);
				updateBias(layer, lastLayer);
				break;
			default:
				break;
			}
		}
	}
	//更新偏置
	private void updateBias(final Layer layer, Layer lastLayer) {
		final double[][][][] errors = layer.getErrors();
		int mapNum = layer.getOutMapNum();

		new ConcurentRunner.TaskManager(mapNum) {
			@Override
			public void process(int start, int end) {
				for (int j = start; j < end; j++) {
					double[][] error = MathUtils.sum(errors, j);
					double deltaBias = MathUtils.sum(error) / batchSize;
					double bias = layer.getBias(j) +ALPHA * deltaBias;
					layer.setBias(j, bias);
				}
			}
		}.start();

	}
	//更新内卷核参数
	private void updateKernels(final Layer layer, final Layer lastLayer) {

		int mapNum = layer.getOutMapNum();
		final int lastMapNum = lastLayer.getOutMapNum();

		new ConcurentRunner.TaskManager(mapNum) {

			@Override
			public void process(int start, int end) {
				for (int j = start; j < end; j++) {
					for (int i = 0; i < lastMapNum; i++) {
						double[][] deltaKernel = null;
						for (int r = 0; r < batchSize; r++) {
							double[][] error = layer.getError(r, j);
							double[][] currmatrix=lastLayer.getMap(r,i);
							if (deltaKernel == null)
								deltaKernel =MathUtils.convnValid(currmatrix,error);
							else {
								deltaKernel = MathUtils.matrixOp(MathUtils.convnValid(currmatrix,error),
										deltaKernel, null, null, MathUtils.plus);
							}
						}
						deltaKernel = MathUtils.matrixOp(deltaKernel, divide_batchSize);
						double[][] kernel = layer.getKernel(i, j);
						deltaKernel = MathUtils.matrixOp(kernel, deltaKernel, null, multiply_alpha,
								MathUtils.plus);
						layer.setKernel(i, j, deltaKernel);
					}
				}

			}
		}.start();

	}
	//设置隐藏层误差
	private void setHiddenLayerErrors() {
		for (int l = layerNum - 2; l > 0; l--) {
			Layer layer = layers.get(l);
			Layer nextLayer = layers.get(l + 1);
			switch (layer.getType()) {
			case samp:
				setSampErrors(layer, nextLayer);
				break;
			case conv:
				setConvErrors(layer, nextLayer);
				break;
			default:
				break;
			}
		}
	}
	//设置（隐藏）池化层误差
	private void setSampErrors(final Layer layer, final Layer nextLayer) {
	//图片数
		int mapNum = layer.getOutMapNum();
		final int nextMapNum = nextLayer.getOutMapNum();

		new ConcurentRunner.TaskManager(mapNum) {

			@Override
			public void process(int start, int end) {
				for (int i = start; i < end; i++) {
					double[][] sum = null;
					for (int j = 0; j < nextMapNum; j++) {
						double[][] currentOutMatrix=layer.getMap(i);
						double[][] nextError = nextLayer.getError(j);
						double[][] kernel = nextLayer.getKernel(i, j);
						if (sum == null)
							sum = MathUtils.convnFull(nextError, MathUtils.rot180(kernel));
						else
							sum = MathUtils.matrixOp(MathUtils.convnFull(nextError, MathUtils.rot180(kernel)), sum,
									null, null, MathUtils.plus);
					}
					layer.setError(i, sum);
				}
			}

		}.start();

	}

	//设置卷化层误差
	private void setConvErrors(final Layer layer, final Layer nextLayer) {

		int mapNum = layer.getOutMapNum();

		new ConcurentRunner.TaskManager(mapNum) {

			@Override
			public void process(int start, int end) {
				for (int m = start; m < end; m++) {
					Layer.Size scale = nextLayer.getScaleSize();
					double[][] nextError = nextLayer.getError(m);
					double[][] map = layer.getMap(m);
					double[][] outMatrix = MathUtils.matrixOp(map, MathUtils.cloneMatrix(map), MathUtils.one_value,
							MathUtils.two_value, MathUtils.multiply);
					outMatrix = MathUtils.matrixOp(outMatrix, MathUtils.kronecker(nextError, scale), null, null,
							MathUtils.multiply);
					layer.setError(m, outMatrix);
				}
			}

		}.start();

	}
	//设置输出层误差
	private boolean setOutLayerErrors(DataSet.Record record) {

		Layer outputLayer = layers.get(layerNum - 1);
		int mapNum = outputLayer.getOutMapNum();

		double[] target = new double[mapNum];
		double[] outmaps = new double[mapNum];
		for (int m = 0; m < mapNum; m++) {
			double[][] outmap = outputLayer.getMap(m);
			outmaps[m] = outmap[0][0];

		}
		int lable = record.getLabel().intValue();
		target[lable] = 1;
		int index=MathUtils.getMaxIndex(outmaps);
		for (int m = 0; m < mapNum; m++) {
			outputLayer.setError(m,0,0,(target[m]-outmaps[m])*(1-outmaps[m]*outmaps[m]));
		}

		return lable == index;
	}

	private void forward(DataSet.Record record) {
		//为输入层赋值
		setInLayerOutput(record);
		//CNN操作
		for (int l = 1; l < layers.size(); l++) {
			Layer layer = layers.get(l);
			Layer lastLayer = layers.get(l - 1);
			switch (layer.getType()) {
			case conv:
				setConvOutput(layer, lastLayer);
				break;
			case samp:
				setSampOutput(layer, lastLayer);
				break;
			case output:
				setConvOutput(layer, lastLayer);
				break;
			default:
				break;
			}
		}
	}

	//设置输入层数据
	private void setInLayerOutput(DataSet.Record record) {
		final Layer inputLayer = layers.get(0);
		final Layer.Size mapSize = inputLayer.getMapSize();
		final double[] attr = record.getAttrs();//获取一行数据
		if (attr.length != mapSize.x * mapSize.y)//没有使用padding层，所以硬指标
			throw new RuntimeException("map");
		for (int i = 0; i < mapSize.x; i++) {
			for (int j = 0; j < mapSize.y; j++) {
				// inputLayer.setMapValue(0, i, j, attr[mapSize.x * i + j]);
				inputLayer.setMapValue(0, i, j, attr[mapSize.y * i + j]);
			}
		}
	}

	//设置卷化层输出
	private void setConvOutput(final Layer layer, final Layer lastLayer) {

		int mapNum = layer.getOutMapNum();
		//前一层的图片数量
		final int lastMapNum = lastLayer.getOutMapNum();

		new ConcurentRunner.TaskManager(mapNum) {

			@Override
			//使用并发，提高效率
			public void process(int start, int end) {
				for (int j = start; j < end; j++) {
					double[][] sum = null;
					for (int i = 0; i < lastMapNum; i++) {
						double[][] lastMap = lastLayer.getMap(i);
						double[][] kernel = layer.getKernel(i, j);
						if (sum == null)
							//生成结果矩阵(卷化)
							sum = MathUtils.convnValid(lastMap, kernel);
						else
							//将几个卷化结果对应相加
							sum = MathUtils.matrixOp(MathUtils.convnValid(lastMap, kernel), sum, null, null,
									MathUtils.plus);
					}
					final double bias = layer.getBias(j);
					//对矩阵各值进行sofaplus函数
					sum = MathUtils.matrixOp(sum, new MathUtils.Operator() {
						private static final long serialVersionUID = 2469461972825890810L;

						@Override
						public double process(double value) {
							return MathUtils.tanh(value + bias);
						}

					});

					layer.setMapValue(j, sum);
				}
			}

		}.start();

	}
	//设置输出层输出
	private void setOutOutput(final Layer layer, final Layer lastLayer) {

		int mapNum = layer.getOutMapNum();
		//前一层的图片数量
		final int lastMapNum = lastLayer.getOutMapNum();

		new ConcurentRunner.TaskManager(mapNum) {

			@Override
			//使用并发，提高效率
			public void process(int start, int end) {
				for (int j = start; j < end; j++) {
					double[][] sum = null;
					for (int i = 0; i < lastMapNum; i++) {
						double[][] lastMap = lastLayer.getMap(i);
						double[][] kernel = layer.getKernel(i, j);
						if (sum == null)
							//生成结果矩阵(卷化)
							sum = MathUtils.convnValid(lastMap, kernel);
						else
							//将几个卷化结果对应相加
							sum = MathUtils.matrixOp(MathUtils.convnValid(lastMap, kernel), sum, null, null,
									MathUtils.plus);
					}
					final double bias = layer.getBias(j);
					//对矩阵各值进行激活函数
					sum = MathUtils.matrixOp(sum,bias);

					layer.setMapValue(j, sum);
				}
			}

		}.start();
	}
	//设置池化数据
	private void setSampOutput(final Layer layer, final Layer lastLayer) {

		int lastMapNum = lastLayer.getOutMapNum();

		new ConcurentRunner.TaskManager(lastMapNum) {

			@Override
			public void process(int start, int end) {
				for (int i = start; i < end; i++) {
					double[][] lastMap = lastLayer.getMap(i);
					Layer.Size scaleSize = layer.getScaleSize();
					double[][] sampMatrix = MathUtils.scaleMatrix(lastMap, scaleSize);
					layer.setMapValue(i, sampMatrix);
				}
			}

		}.start();

	}

	//开始设置（）各层初始化
	public void setup(int batchSize) {

		Layer inputLayer = layers.get(0);//获取输入层
		inputLayer.initOutmaps(batchSize);

		for (int i = 1; i < layers.size(); i++) {
			Layer layer = layers.get(i);
			Layer frontLayer = layers.get(i - 1);
			//前一层图片数目
			int frontMapNum = frontLayer.getOutMapNum();

			switch (layer.getType()) {
			case input:
				break;
			case conv:
				//设置输出图大小
				layer.setMapSize(frontLayer.getMapSize().subtract(layer.getKernelSize(), 1));

				layer.initKernel(frontMapNum);
				//偏置值设置为0
				layer.initBias(frontMapNum);
				//
				layer.initErros(batchSize);
				layer.initOutmaps(batchSize);
				break;
			case samp:
				layer.setOutMapNum(frontMapNum);
				layer.setMapSize(frontLayer.getMapSize().divide(layer.getScaleSize()));
				layer.initBias(frontMapNum);
				layer.initErros(batchSize);
				layer.initOutmaps(batchSize);
				break;
			case output:
				layer.initOutputKerkel(frontMapNum, frontLayer.getMapSize());
				layer.initBias(frontMapNum);
				layer.initErros(batchSize);
				layer.initOutmaps(batchSize);
				break;
			}
		}
	}

	public static class LayerBuilder {
     //创建层次集合
		private List<Layer> mLayers;

		public LayerBuilder() {
			mLayers = new ArrayList<Layer>();
		}

		public LayerBuilder(Layer layer) {
			this();
			mLayers.add(layer);
		}

		public LayerBuilder addLayer(Layer layer) {
			mLayers.add(layer);
			return this;
		}

	}

	public void saveModel(String fileName) {
		try (ObjectOutputStream oos = new ObjectOutputStream(new FileOutputStream(fileName));) {
			oos.writeObject(this);
			oos.flush();
		} catch (IOException e) {
			logger.error("IOException:{}", e);
			throw new RuntimeException(e);
		}
	}

	public static CNN loadModel(String fileName) {
		try (ObjectInputStream in = new ObjectInputStream(new FileInputStream(fileName));) {
			CNN cnn = (CNN) in.readObject();
			return cnn;
		} catch (IOException | ClassNotFoundException e) {
			logger.error("IOException or ClassNotFoundException:{}", e);
			throw new RuntimeException(e);
		}
	}

}
