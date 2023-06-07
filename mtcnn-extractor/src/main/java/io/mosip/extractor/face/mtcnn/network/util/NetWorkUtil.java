package io.mosip.extractor.face.mtcnn.network.util;

import java.util.ArrayList;
import java.util.Collections;
import java.util.Comparator;

import org.opencv.core.CvType;
import org.opencv.core.Mat;

import io.mosip.extractor.constant.ExtractorErrorCode;
import io.mosip.extractor.constant.MtcnnConstant;
import io.mosip.extractor.exception.ExtractorException;
import io.mosip.extractor.face.mtcnn.dto.BoundingBox;
import io.mosip.extractor.face.mtcnn.dto.OrderScore;
import io.mosip.extractor.face.mtcnn.dto.PBox;
import io.mosip.extractor.face.mtcnn.dto.PRelu;
import io.mosip.extractor.face.mtcnn.dto.Weight;
import io.mosip.extractor.face.mtcnn.util.Util;

public class NetWorkUtil {
	public static void addBias(PBox pBox, ArrayList<Float> pBias) {
		ExtractorErrorCode errorCode = null;
		if (pBox == null || pBox.getPData() == null) {
			errorCode = ExtractorErrorCode.RELU_FEATURE_NULL_EXCEPTION;
			throw new ExtractorException (errorCode.getErrorCode(), errorCode.getErrorMessage());
		}
		if (pBias == null) {
			errorCode = ExtractorErrorCode.RELU_BIAS_NULL_EXCEPTION;
			throw new ExtractorException (errorCode.getErrorCode(), errorCode.getErrorMessage());
		}

		int opCount = 0;
		int pbCount = 0;

		long distance = pBox.getWidth() * pBox.getHeight();
		for (int channel = 0; channel < pBox.getChannel(); channel++) {
			for (int col = 0; col < distance; col++) {
				pBox.getPData().set(opCount, pBox.getPData().get(opCount) + pBias.get(pbCount));
				opCount++;
			}
			pbCount++;
		}
	}

	public static void maxPooling(PBox pBox, PBox matrix, int kernelSize, int stride) {
		ExtractorErrorCode errorCode = null;
		if (pBox == null || pBox.getPData() == null) {
			errorCode = ExtractorErrorCode.MAX_POOLING_PROPOSE_BOX_NULL_EXCEPTION;
			throw new ExtractorException (errorCode.getErrorCode(), errorCode.getErrorMessage());
		}
		
		int pCount = 0;
		int pInCount = 0;
		int pTempCount = 0;
		float maxNum = 0;
		if ((pBox.getWidth() - kernelSize) % stride == 0 && (pBox.getHeight() - kernelSize) % stride == 0) {
			for (int row = 0; row < matrix.getHeight(); row++) {
				for (int col = 0; col < matrix.getWidth(); col++) {
					pInCount = row * stride * pBox.getWidth() + col * stride;
					for (int channel = 0; channel < pBox.getChannel(); channel++) {
						pTempCount = pInCount + channel * pBox.getHeight() * pBox.getWidth();
						maxNum = pBox.getPData().get(pTempCount);
						for (int kernelRow = 0; kernelRow < kernelSize; kernelRow++) {
							for (int i = 0; i < kernelSize; i++) {
								if (maxNum < pBox.getPData().get(pTempCount + i + kernelRow * pBox.getWidth())) {
									maxNum = pBox.getPData().get(pTempCount + i + kernelRow * pBox.getWidth());
								}
							}
						}
						matrix.getPData().set(pCount + channel * matrix.getHeight() * matrix.getWidth(), maxNum);
					}
					pCount++;
				}
			}
		} else {
			int diffh = 0, diffw = 0;
			for (int channel = 0; channel < pBox.getChannel(); channel++) {
				pInCount = channel * pBox.getHeight() * pBox.getWidth();
				for (int row = 0; row < matrix.getHeight(); row++) {
					for (int col = 0; col < matrix.getWidth(); col++) {
						pTempCount = pInCount + row * stride * pBox.getWidth() + col * stride;
						maxNum = pBox.getPData().get(pTempCount);
						diffh = row * stride - pBox.getHeight() + 1;
						diffw = col * stride - pBox.getHeight() + 1;
						for (int kernelRow = 0; kernelRow < kernelSize; kernelRow++) {
							if ((kernelRow + diffh) > 0)
								break;
							for (int i = 0; i < kernelSize; i++) {
								if ((i + diffw) > 0)
									break;
								if ((pTempCount + i + kernelRow * pBox.getWidth()) >= pBox.getPData().size())
									break;
								if (maxNum < pBox.getPData().get(pTempCount + i + kernelRow * pBox.getWidth())) {
									maxNum = pBox.getPData().get(pTempCount + i + kernelRow * pBox.getWidth());
								}
							}
						}
						matrix.getPData().set(pCount++, maxNum);
					}
				}
			}
		}
	}

	public static void maxPoolingInit(PBox pBox, PBox matrix, int kernelSize, int stride) {
		matrix.setWidth((int) Math.ceil((float) (pBox.getWidth() - kernelSize) / stride + 1));
		matrix.setHeight((int) Math.ceil((float) (pBox.getHeight() - kernelSize) / stride + 1));
		matrix.setChannel(pBox.getChannel());
		long byteLength = matrix.getChannel() * matrix.getWidth() * matrix.getHeight();
		matrix.setPData(new ArrayList<Float>());
		for (int i = 0; i < byteLength; i++) {
			matrix.getPData().add(0.0F);
		}
	}

	public static void prelu(PBox pBox, ArrayList<Float> pBias, ArrayList<Float> prelu_gmma) {
		ExtractorErrorCode errorCode = null;
		if (pBox != null && pBox.getPData() == null) {
			errorCode = ExtractorErrorCode.RELU_FEATURE_NULL_EXCEPTION;
			throw new ExtractorException (errorCode.getErrorCode(), errorCode.getErrorMessage());
		}
		if (pBias == null) {
			errorCode = ExtractorErrorCode.RELU_BIAS_NULL_EXCEPTION;
			throw new ExtractorException (errorCode.getErrorCode(), errorCode.getErrorMessage());
		}

		int opCount = 0;
		int pbCount = 0;
		int pgCount = 0;
		long dis = pBox.getWidth() * pBox.getHeight();
		for (int channel = 0; channel < pBox.getChannel(); channel++) {
			for (int col = 0; col < dis; col++) {
				pBox.getPData().set(opCount, pBox.getPData().get(opCount) + pBias.get(pbCount));
				if (pBox.getPData().get(opCount) > 0) {
					pBox.getPData().set(opCount, pBox.getPData().get(opCount));
				} else {
					pBox.getPData().set(opCount, pBox.getPData().get(opCount) * prelu_gmma.get(pgCount));
				}
				opCount++;
			}
			pbCount++;
			pgCount++;
		}
	}

	public static void fullConnect(Weight weight, PBox pBox, PBox outPBox) {
		ExtractorErrorCode errorCode = null;
		if (pBox != null && pBox.getPData() == null) {
			errorCode = ExtractorErrorCode.FC_FEATURE_NULL_EXCEPTION;
			throw new ExtractorException (errorCode.getErrorCode(), errorCode.getErrorMessage());
		}
		if (weight != null && weight.getPData() == null) {
			errorCode = ExtractorErrorCode.FC_WEIGHT_NULL_EXCEPTION;
			throw new ExtractorException (errorCode.getErrorCode(), errorCode.getErrorMessage());
		}

		Util.gemvCpu(weight.getSelfChannel(), weight.getLastChannel(), 1, weight.getPData(), weight.getLastChannel(),
				pBox.getPData(), 1, 0, outPBox.getPData(), 1);
	}

	public static void fullConnectInit(Weight weight, PBox outputBox) {
		outputBox.setChannel(weight.getSelfChannel());
		outputBox.setWidth(1);
		outputBox.setHeight(1);

		long byteLength = weight.getSelfChannel();
		outputBox.setPData(new ArrayList<Float>());
		for (int i = 0; i < byteLength; i++) {
			outputBox.getPData().add(0.0F);
		}
	}

	public static void convolution(Weight weight, PBox pBox, PBox outpBox, PBox matrix) {
		ExtractorErrorCode errorCode = null;
		if (pBox != null && pBox.getPData() == null) {
			errorCode = ExtractorErrorCode.FEATURE_NULL_EXCEPTION;
			throw new ExtractorException (errorCode.getErrorCode(), errorCode.getErrorMessage());
		}
		if (weight != null && weight.getPData() == null) {
			errorCode = ExtractorErrorCode.WEIGHT_NULL_EXCEPTION;
			throw new ExtractorException (errorCode.getErrorCode(), errorCode.getErrorMessage());
		}

		Util.gemmCpu(weight.getSelfChannel(), matrix.getHeight(), matrix.getWidth(), 1, weight.getPData(),
				matrix.getWidth(), matrix.getPData(), matrix.getWidth(), 0, outpBox.getPData(), matrix.getHeight());
	}

	public static void convolutionInit(Weight weight, PBox pbox, PBox outputBox, PBox matrix) {
		outputBox.setChannel(weight.getSelfChannel());
		outputBox.setWidth((pbox.getWidth() - weight.getKernelSize()) / weight.getStride() + 1);
		outputBox.setHeight((pbox.getHeight() - weight.getKernelSize()) / weight.getStride() + 1);
		long byteLength = weight.getSelfChannel() * matrix.getHeight();
		outputBox.setPData(new ArrayList<Float>());

		for (int i = 0; i < byteLength; i++) {
			outputBox.getPData().add(0.0F);
		}
	}

	public static void feature2Matrix(PBox pBox, PBox matrixBox, Weight weight) {
		ExtractorErrorCode errorCode = null;
		if (pBox != null && pBox.getPData() == null) {
			errorCode = ExtractorErrorCode.FEATURE_MATRIX_BOX_NULL_EXCEPTION;
			throw new ExtractorException (errorCode.getErrorCode(), errorCode.getErrorMessage());
		}
		
		int kernelSize = weight.getKernelSize();
		int stride = weight.getStride();
		int w_out = (pBox.getWidth() - kernelSize) / stride + 1;
		int h_out = (pBox.getHeight() - kernelSize) / stride + 1;

		int m_count = 0;
		int pIn_count = 0;
		int pTemp_count = 0;
		for (int row = 0; row < h_out; row++) {
			for (int col = 0; col < w_out; col++) {
				pIn_count = row * stride * pBox.getWidth() + col * stride;

				for (int channel = 0; channel < pBox.getChannel(); channel++) {
					pTemp_count = pIn_count + channel * pBox.getHeight() * pBox.getWidth();
					for (int kernelRow = 0; kernelRow < kernelSize; kernelRow++) {
						for (int n = 0; n < kernelSize; n++) {
							matrixBox.getPData().set(n + m_count, pBox.getPData().get(pTemp_count + n));
						}
						m_count += kernelSize;
						pTemp_count += pBox.getWidth();
					}
				}
			}
		}
	}

	public static void feature2MatrixInit(PBox pbox, PBox matrixBox, Weight weight) {
		int kernelSize = weight.getKernelSize();
		int stride = weight.getStride();
		int w_out = (pbox.getWidth() - kernelSize) / stride + 1;
		int h_out = (pbox.getHeight() - kernelSize) / stride + 1;

		matrixBox.setWidth(pbox.getChannel() * kernelSize * kernelSize);
		matrixBox.setHeight(w_out * h_out);
		matrixBox.setChannel(1);

		long byteLength = matrixBox.getWidth() * matrixBox.getHeight();
		matrixBox.setPData(new ArrayList<Float>());
		for (int i = 0; i < byteLength; i++) {
			matrixBox.getPData().add(0.0F);
		}
	}

	public static void image2Matrix(Mat image, PBox pbox) {
		ExtractorErrorCode errorCode = null;
		if (image == null || image.empty() || (image.type() != CvType.CV_8UC3)) {
			errorCode = ExtractorErrorCode.IMAGE_MATRIX_EMPTY_OR_IMAGE_TYPE_WRONG_EXCEPTION;
			throw new ExtractorException (errorCode.getErrorCode(), errorCode.getErrorMessage());
		}
		
		if (pbox == null || pbox.getPData() == null) {
			errorCode = ExtractorErrorCode.IMAGE_MATRIX_BOUNDING_BOX_NULL_EXCEPTION;
			throw new ExtractorException (errorCode.getErrorCode(), errorCode.getErrorMessage());
		}

		int p_count = 0;
		double[] pixel = new double[3];
		for (int rowI = 0; rowI < image.rows(); rowI++) {
			for (int colK = 0; colK < image.cols(); colK++) {
				pixel = image.get(rowI, colK).clone();
				pbox.getPData().set(p_count, (float) ((pixel[0] - 127.5) * 0.0078125));
				pbox.getPData().set(p_count + image.rows() * image.cols(), (float) ((pixel[1] - 127.5) * 0.0078125));
				pbox.getPData().set(p_count + 2 * image.rows() * image.cols(),
						(float) ((pixel[2] - 127.5) * 0.0078125));
				p_count++;
			}
		}
	}

	public static void image2MatrixInit(Mat image, PBox pbox) {
		ExtractorErrorCode errorCode = null;
		if (image == null || image.empty() || (image.type() != CvType.CV_8UC3)) {
			errorCode = ExtractorErrorCode.IMAGE_MATRIX_EMPTY_OR_IMAGE_TYPE_WRONG_EXCEPTION;
			throw new ExtractorException (errorCode.getErrorCode(), errorCode.getErrorMessage());
		}
		pbox.setChannel(image.channels());
		pbox.setHeight(image.rows());
		pbox.setWidth(image.cols());
		long byteLength = pbox.getChannel() * pbox.getHeight() * pbox.getWidth();
		pbox.setPData(new ArrayList<Float>());
		for (int i = 0; i < byteLength; i++) {
			pbox.getPData().add(0.0F);
		}
	}

	public static long initConvAndFc(Weight weight, int selfChannel, int lastChannel, int kernelSize, int stride,
			int pad) {
		weight.setSelfChannel(selfChannel);
		weight.setLastChannel(lastChannel);
		weight.setKernelSize(kernelSize);
		weight.setStride(stride);
		weight.setPad(pad);

		// initial pbias
		weight.setPBias(new ArrayList<Float>());
		for (int i = 0; i < selfChannel; i++) {
			weight.getPBias().add(0.0F);
		}

		// initial pdata
		long byteLength = weight.getSelfChannel() * weight.getLastChannel() * weight.getKernelSize()
				* weight.getKernelSize();
		weight.setPData(new ArrayList<Float>());
		for (int i = 0; i < byteLength; i++) {
			weight.getPData().add(0.0F);
		}

		return byteLength;
	}

	public static void initpRelu(PRelu prelu, int width) {
		prelu.setWidth(width);
		prelu.setPData(new ArrayList<Float>());
		for (int i = 0; i < width; i++) {
			prelu.getPData().add(0.0F);
		}
	}

	public static void softmax(PBox pBox) {
		ExtractorErrorCode errorCode = null;
		if (pBox == null || pBox.getPData() == null) {
			errorCode = ExtractorErrorCode.SOFT_MAX_BOUNDING_BOX_NULL_EXCEPTION;
			throw new ExtractorException (errorCode.getErrorCode(), errorCode.getErrorMessage());
		}

		long p2DCount = 0;
		long p3DCount = 0;
		long mapSize = pBox.getWidth() * pBox.getHeight();
		float eleSum = 0;
		for (int row = 0; row < pBox.getHeight(); row++) {
			for (int col = 0; col < pBox.getWidth(); col++) {
				eleSum = 0;
				for (int channel = 0; channel < pBox.getChannel(); channel++) {
					p3DCount = p2DCount + channel * mapSize;
					pBox.getPData().set((int) p3DCount, (float) Math.exp(pBox.getPData().get((int) p3DCount)));
					eleSum += pBox.getPData().get((int) p3DCount);
				}
				for (int channel = 0; channel < pBox.getChannel(); channel++) {
					p3DCount = p2DCount + channel * mapSize;
					pBox.getPData().set((int) p3DCount, pBox.getPData().get((int) p3DCount) / eleSum);
				}
				p2DCount++;
			}
		}
	}

	public static void nms(ArrayList<BoundingBox> boundingBox, ArrayList<OrderScore> boundingBoxScore,
			float overlapThreshold, String modelName) {
		if (boundingBox.isEmpty()) {
			return;
		}
		ArrayList<Integer> heros = new ArrayList<Integer>();
		// sort the score(small -> big)
		Collections.sort(boundingBoxScore, new Comparator<OrderScore>() {
			@Override
			public int compare(OrderScore lsh, OrderScore rsh) {
				if (lsh.getScore() > rsh.getScore()) {
					return 1;
				} else if (lsh.getScore() < rsh.getScore()) {
					return -1;
				} else {
					return 0;
				}
			}
		});

		int order = 0;
		float IOU = 0;
		float maxX = 0;
		float maxY = 0;
		float minX = 0;
		float minY = 0;
		while (boundingBoxScore.size() > 0) {
			order = boundingBoxScore.get(boundingBoxScore.size() - 1).getOriOrder();
			boundingBoxScore.remove(boundingBoxScore.size() - 1);
			if (order < 0) {
				continue;
			}
			heros.add(order);
			boundingBox.get(order).setExist(false);// delete it

			for (int num = 0; num < boundingBox.size(); num++) {
				if (boundingBox.get(num).isExist()) {
					// the iou
					maxX = (boundingBox.get(num).getX1() > boundingBox.get(order).getX1())
							? boundingBox.get(num).getX1()
							: boundingBox.get(order).getX1();
					maxY = (boundingBox.get(num).getY1() > boundingBox.get(order).getY1())
							? boundingBox.get(num).getY1()
							: boundingBox.get(order).getY1();
					minX = (boundingBox.get(num).getX2() < boundingBox.get(order).getX2())
							? boundingBox.get(num).getX2()
							: boundingBox.get(order).getX2();
					minY = (boundingBox.get(num).getY2() < boundingBox.get(order).getY2())
							? boundingBox.get(num).getY2()
							: boundingBox.get(order).getY2();
					// maxX1 and maxY1 reuse
					maxX = ((minX - maxX + 1) > 0) ? (minX - maxX + 1) : 0;
					maxY = ((minY - maxY + 1) > 0) ? (minY - maxY + 1) : 0;
					// IOU reuse for the area of two boundingbox
					IOU = maxX * maxY;
					if (modelName.compareTo(MtcnnConstant.BOUNDINGBOX_MODEL_UNION) == 0) {
						IOU = IOU / (boundingBox.get(num).getArea() + boundingBox.get(order).getArea() - IOU);
					} else if (modelName.compareTo(MtcnnConstant.BOUNDINGBOX_MODEL_MINIMUM) == 0) {
						IOU = IOU / ((boundingBox.get(num).getArea() < boundingBox.get(order).getArea())
								? boundingBox.get(num).getArea()
								: boundingBox.get(order).getArea());
					}
					if (IOU > overlapThreshold) {
						boundingBox.get(num).setExist(false);
						for (int k = 0; k < boundingBoxScore.size(); k++) {
							if (boundingBoxScore.get(k).getOriOrder() == num) {
								boundingBoxScore.get(k).setOriOrder(-1);
								break;
							}
						}
					}
				}
			}
		}
		for (int i = 0; i < heros.size(); i++) {
			boundingBox.get(heros.get(i)).setExist(true);
		}
	}

	public static void refineAndSquareBbox(ArrayList<BoundingBox> vectorBoundingBox, int height, int width) {
		ExtractorErrorCode errorCode = null;
		if (vectorBoundingBox == null || vectorBoundingBox.isEmpty()) {
			errorCode = ExtractorErrorCode.REFINE_NET_BOUNDING_BOX_NULL_OR_EMPTY_EXCEPTION;
			throw new ExtractorException (errorCode.getErrorCode(), errorCode.getErrorMessage());
		}

		float bbw = 0, bbh = 0, maxSide = 0;
		float h = 0, w = 0;
		float x1 = 0, y1 = 0, x2 = 0, y2 = 0;
		for (int i = 0; i < vectorBoundingBox.size(); i++) {
			if (vectorBoundingBox.get(i).isExist()) {
				bbh = vectorBoundingBox.get(i).getX2() - vectorBoundingBox.get(i).getX1() + 1;
				bbw = vectorBoundingBox.get(i).getY2() - vectorBoundingBox.get(i).getY1() + 1;
				x1 = vectorBoundingBox.get(i).getX1() + vectorBoundingBox.get(i).getRegreCoordinates()[1] * bbh;
				y1 = vectorBoundingBox.get(i).getY1() + vectorBoundingBox.get(i).getRegreCoordinates()[0] * bbw;
				x2 = vectorBoundingBox.get(i).getX2() + vectorBoundingBox.get(i).getRegreCoordinates()[3] * bbh;
				y2 = vectorBoundingBox.get(i).getY2() + vectorBoundingBox.get(i).getRegreCoordinates()[2] * bbw;

				h = x2 - x1 + 1;
				w = y2 - y1 + 1;

				maxSide = (h > w) ? h : w;
				x1 = x1 + h * 0.5F - maxSide * 0.5F;
				y1 = y1 + w * 0.5F - maxSide * 0.5F;
				vectorBoundingBox.get(i).setX2(Math.round(x1 + maxSide - 1));
				vectorBoundingBox.get(i).setY2(Math.round(y1 + maxSide - 1));
				vectorBoundingBox.get(i).setX1(Math.round(x1));
				vectorBoundingBox.get(i).setY1(Math.round(y1));

				// boundary check
				if (vectorBoundingBox.get(i).getX1() < 0) {
					vectorBoundingBox.get(i).setX1(0);
				}
				if (vectorBoundingBox.get(i).getY1() < 0) {
					vectorBoundingBox.get(i).setY1(0);
				}
				if (vectorBoundingBox.get(i).getX2() > height) {
					vectorBoundingBox.get(i).setX2(height - 1);
				}
				if (vectorBoundingBox.get(i).getY2() > width) {
					vectorBoundingBox.get(i).setY2(width - 1);
				}

				vectorBoundingBox.get(i).setArea((vectorBoundingBox.get(i).getX2() - vectorBoundingBox.get(i).getX1())
						* (vectorBoundingBox.get(i).getY2() - vectorBoundingBox.get(i).getY1()));
			}
		}
	}
}