package cnn.utils;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.concurrent.CountDownLatch;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;

public class ConcurentRunner {

	private static Logger logger = LoggerFactory.getLogger(ConcurentRunner.class);
	//创建线程池
	private static final ExecutorService exec;
	//虚拟机cpu数目
	public static final int cpuNum;

	static {
		cpuNum = Runtime.getRuntime().availableProcessors();
		// cpuNum = 1;
		logger.info("cpuNum:{}", cpuNum);
		exec = Executors.newFixedThreadPool(cpuNum);
	}

	public static void run(Runnable task) {
		exec.execute(task);
	}

	public static void stop() {
		exec.shutdown();
	}


	public abstract static class TaskManager {

		private int workLength;

		public TaskManager(int workLength) {
			this.workLength = workLength;
		}

		public void start() {
			int runCpu = cpuNum < workLength ? cpuNum : 1;
			final CountDownLatch gate = new CountDownLatch(runCpu);
			int fregLength = (workLength + runCpu - 1) / runCpu;
			for (int cpu = 0; cpu < runCpu; cpu++) {
				final int start = cpu * fregLength;
				int tmp = (cpu + 1) * fregLength;
				final int end = tmp <= workLength ? tmp : workLength;
				Runnable task = new Runnable() {

					@Override
					public void run() {
						process(start, end);
						//通知主线程
						gate.countDown();
					}

				};
				ConcurentRunner.run(task);
			}
			try {
				gate.await();
			} catch (InterruptedException e) {
				e.printStackTrace();
				throw new RuntimeException(e);
			}
		}

		public abstract void process(int start, int end);

	}

}
