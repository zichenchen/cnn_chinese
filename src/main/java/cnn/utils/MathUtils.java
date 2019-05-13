package cnn.utils;

import cnn.core.Layer;
import java.io.Serializable;
import java.util.Arrays;
import java.util.HashSet;
import java.util.Random;
import java.util.Set;

public class MathUtils {

	public interface Operator extends Serializable {
		public double process(double value);
	}

	public static final Operator one_value = new Operator() {

		private static final long serialVersionUID = 3752139491940330714L;

		@Override
		public double process(double value) {
			return 1 - value;
		}

	};
	public static final Operator two_value = new Operator() {

		private static final long serialVersionUID = 3752139491940330714L;

		@Override
		public double process(double value) {
			return 1 + value;
		}

	};

	public static final Operator digmod = new Operator() {

		private static final long serialVersionUID = -1952718905019847589L;

		@Override
		public double process(double value) {
			return 1 / (1 + Math.pow(Math.E, -value));
		}

	};

	interface OperatorOnTwo extends Serializable {
		public double process(double a, double b);
	}

	public static final OperatorOnTwo plus = new OperatorOnTwo() {

		private static final long serialVersionUID = -6298144029766839945L;

		@Override
		public double process(double a, double b) {
			return a + b;
		}

	};

	public static OperatorOnTwo multiply = new OperatorOnTwo() {

		private static final long serialVersionUID = -7053767821858820698L;

		@Override
		public double process(double a, double b) {
			return a * b;
		}

	};

	public static OperatorOnTwo minus = new OperatorOnTwo() {

		private static final long serialVersionUID = 7346065545555093912L;

		@Override
		public double process(double a, double b) {
			return a - b;
		}

	};

	public static void printMatrix(double[][] matrix) {
		for (int i = 0; i < matrix.length; i++) {
			String line = Arrays.toString(matrix[i]);
			line = line.replaceAll(", ", "\t");
			System.out.println(line);
		}
		System.out.println();
	}

	public static double[][] rot180(double[][] matrix) {
		matrix = cloneMatrix(matrix);
		int m = matrix.length;
		int n = matrix[0].length;
		for (int i = 0; i < m; i++) {
			for (int j = 0; j < n / 2; j++) {
				double tmp = matrix[i][j];
				matrix[i][j] = matrix[i][n - 1 - j];
				matrix[i][n - 1 - j] = tmp;
			}
		}
		for (int j = 0; j < n; j++) {
			for (int i = 0; i < m / 2; i++) {
				double tmp = matrix[i][j];
				matrix[i][j] = matrix[m - 1 - i][j];
				matrix[m - 1 - i][j] = tmp;
			}
		}
		return matrix;
	}

	private static Random r = new Random(1);

	public static double[][] randomMatrix(int x, int y, boolean b) {
		double[][] matrix = new double[x][y];
		//		int tag = 1;
		for (int i = 0; i < x; i++) {
			for (int j = 0; j < y; j++) {
				// [-0.05,0.05)
				//修改0.05为0.5
				matrix[i][j] = (r.nextDouble() - 0.5) / 10;
				//				matrix[i][j] = tag * 0.5;
				//				if (b)
				//					matrix[i][j] *= 1.0*(i + j + 2) / (i + 1) / (j + 1);
				//				tag *= -1;
			}
		}
		 //printMatrix(matrix);
		return matrix;
	}

	public static double[] randomArray(int len) {
		double[] data = new double[len];
		for (int i = 0; i < len; i++) {
			// data[i] = r.nextDouble() / 10 - 0.05;
			data[i] = 0;
		}
		return data;
	}

	public static int[] randomPerm(int size, int batchSize) {
		Set<Integer> set = new HashSet<Integer>();
		//添加一个0-train.size的随机数
		while (set.size() < batchSize) {
			set.add(r.nextInt(size));
		}
		int[] randPerm = new int[batchSize];
		int i = 0;
		for (Integer value : set)
			randPerm[i++] = value;
		return randPerm;
	}
	//复制一份
	public static double[][] cloneMatrix(final double[][] matrix) {
		final int m = matrix.length;
		int n = matrix[0].length;
		final double[][] outMatrix = new double[m][n];

		for (int i = 0; i < m; i++) {
			for (int j = 0; j < n; j++) {
				outMatrix[i][j] = matrix[i][j];
			}
		}
		return outMatrix;
	}
    //参数（卷化结果，数据操作）
	public static double[][] matrixOp(final double[][] ma, Operator operator) {
		final int m = ma.length;
		int n = ma[0].length;
		for (int i = 0; i < m; i++) {
			for (int j = 0; j < n; j++) {
				ma[i][j] = operator.process(ma[i][j]);
			}
		}
		return ma;
	}
   //sofamax函数
	public static double[][] matrixOp(final double[][] ma,double bias) {
		final int m = ma.length;
		int n = ma[0].length;
		double sum=0;
		for (int i = 0; i < m; i++) {
			for (int j = 0; j < n; j++) {
				sum+=Math.pow(Math.E,ma[i][j]+bias);
			}
		}
		for (int i = 0; i < m; i++) {
			for (int j = 0; j < n; j++) {
				ma[i][j] = (bias+ma[i][j])/sum;
			}
		}
		return ma;
	}
    //参数（卷化结果，sum,null,null,加操作）
	public static double[][] matrixOp(double[][] ma, double[][] mb, final Operator operatorA,
			final Operator operatorB, OperatorOnTwo operator) {
		final int m = ma.length;
		int n = ma[0].length;
		if (m != mb.length || n != mb[0].length)
			throw new RuntimeException("ma.length:" + ma.length + "  mb.length:" + mb.length);

		for (int i = 0; i < m; i++) {
			for (int j = 0; j < n; j++) {
				double a = ma[i][j];
				if (operatorA != null)
					a = operatorA.process(a);
				double b = mb[i][j];
				if (operatorB != null)
					b = operatorB.process(b);

				mb[i][j] = operator.process(a, b);
			}
		}
		return mb;
	}

	public static double[][] kronecker(final double[][] matrix, final Layer.Size scale) {
		final int m = matrix.length;
		int n = matrix[0].length;
		final double[][] outMatrix = new double[m * scale.x][n * scale.y];

		for (int i = 0; i < m; i++) {
			for (int j = 0; j < n; j++) {
				for (int ki = i * scale.x; ki < (i + 1) * scale.x; ki++) {
					for (int kj = j * scale.y; kj < (j + 1) * scale.y; kj++) {
						outMatrix[ki][kj] = matrix[i][j];
					}
				}
			}
		}
		return outMatrix;
	}
    //输出池化后矩阵
	public static double[][] scaleMatrix(final double[][] matrix, final Layer.Size scale) {
		int m = matrix.length;
		int n = matrix[0].length;
		final int sm = m / scale.x;
		final int sn = n / scale.y;
		final double[][] outMatrix = new double[sm][sn];
		if (sm * scale.x != m || sn * scale.y != n)
			throw new RuntimeException("scale matrix");
		final int size = scale.x * scale.y;
		for (int i = 0; i < sm; i++) {
			for (int j = 0; j < sn; j++) {
				double sum = 0.0;
				//池化：求平均
				for (int si = i * scale.x; si < (i + 1) * scale.x; si++) {
					for (int sj = j * scale.y; sj < (j + 1) * scale.y; sj++) {
						sum += matrix[si][sj];
					}
				}
				outMatrix[i][j] = sum / size;
			}
		}
		return outMatrix;
	}
	//
	public static double[][] convnFull(double[][] matrix, final double[][] kernel) {
		int m = matrix.length;
		int n = matrix[0].length;
		final int km = kernel.length;
		final int kn = kernel[0].length;
		final double[][] extendMatrix = new double[m + 2 * (km - 1)][n + 2 * (kn - 1)];
		for (int i = 0; i < m; i++) {
			for (int j = 0; j < n; j++)
				extendMatrix[i + km - 1][j + kn - 1] = matrix[i][j];
		}
		return convnValid(extendMatrix, kernel);
	}
	//tanh已知池化层求上一层
	public static double[][] pooltanh(double[][] nexterror, Layer.Size scale){
		double[][] outresult=new double[nexterror.length*scale.y][nexterror[0].length*scale.x];
		for (int i = 0; i <nexterror.length ; i++) {
			for (int j = 0; j <nexterror[0].length ; j++) {
				outresult[i*2][j*2]=nexterror[i][j];
				outresult[i*2+1][j*2]=nexterror[i][j];
				outresult[i*2][j*2+1]=nexterror[i][j];
				outresult[i*2+1][j*2+1]=nexterror[i][j];
			}
		}
		return outresult;
	}
	//
public  static  double[][] hadamardMatrix(double[][] x1,double[][] x2){
	final double[][] result=new double[x1.length][x1[0].length];
	for(int i=0;i<result.length;i++){
		for (int j = 0; j <result[0].length ; j++) {
			result[i][j]=x1[i][j]*x2[i][j];
		}
	}
	return result;
}

	//tanh函数的求导
	public  static  double[][] derivativetanh(double[][] matrix){
		final  double[][] derivativeMatrix=new double[matrix.length][matrix[0].length];
		for (int i = 0; i <matrix.length ; i++) {
			for (int j = 0; j <matrix[0].length ; j++) {
				derivativeMatrix[i][j]=1-matrix[i][j]*matrix[i][j];
			}
		}
		return derivativeMatrix;
	}

	//tanh已知卷化层求上一层
	public static double[][] convntanh(double[][] nexterror, final double[][] kernel,double[][] currentOutMatrix) {
		int m = nexterror.length;
		int n = nexterror[0].length;
		final int km = kernel.length;
		final int kn = kernel[0].length;
		final double[][] extendMatrix = new double[m + 2 * (km - 1)][n + 2 * (kn - 1)];
		for (int i = 0; i < m; i++) {
			for (int j = 0; j < n; j++)
				extendMatrix[i + km - 1][j + kn - 1] = nexterror[i][j];
		}
		double[][] convnresult=convnValid(extendMatrix, kernel);
		double [][] derivativeMatrix=derivativetanh(currentOutMatrix);
		return hadamardMatrix(convnresult,derivativeMatrix);
	}
    //输出卷化后结果矩阵
	public static double[][] convnValid(final double[][] matrix, double[][] kernel) {
		//		kernel = rot180(kernel);
		int m = matrix.length;
		int n = matrix[0].length;
		final int km = kernel.length;
		final int kn = kernel[0].length;
		int kns = n - kn + 1;
		final int kms = m - km + 1;
		final double[][] outMatrix = new double[kms][kns];

		for (int i = 0; i < kms; i++) {
			for (int j = 0; j < kns; j++) {
				double sum = 0.0;
				for (int ki = 0; ki < km; ki++) {
					for (int kj = 0; kj < kn; kj++)
						sum += matrix[i + ki][j + kj] * kernel[ki][kj];
				}
				outMatrix[i][j] = sum;

			}
		}
		return outMatrix;
	}
    //sigmod函数
	public static double sigmod(double x) {
		return 1 / (1 + Math.pow(Math.E, -x));
	}

	public  static double ReLU(double x){
		return Math.max(0.0,x);
	}

	public static  double sofaplus(double x) { return Math.log(1+Math.pow(Math.E,x));}

	public static  double tanh(double x) { return (Math.exp(x)-Math.exp(-x))/(Math.exp(x)+Math.exp(-x));}

	public static double sum(double[][] error) {
		int m = error.length;
		int n = error[0].length;
		double sum = 0.0;
		for (int i = 0; i < m; i++) {
			for (int j = 0; j < n; j++) {
				sum += error[i][j];
			}
		}
		return sum;
	}

	public static double[][] sum(double[][][][] errors, int j) {
		int m = errors[0][j].length;
		int n = errors[0][j][0].length;
		double[][] result = new double[m][n];
		for (int mi = 0; mi < m; mi++) {
			for (int nj = 0; nj < n; nj++) {
				double sum = 0;
				for (int i = 0; i < errors.length; i++)
					sum += errors[i][j][mi][nj];
				result[mi][nj] = sum;
			}
		}
		return result;
	}

	public static int getMaxIndex(double[] out) {
		double max = out[0];
		int index = 0;
		for (int i = 1; i < out.length; i++)
			if (out[i] > max) {
				max = out[i];
				index = i;
			}
		return index;
	}



}
