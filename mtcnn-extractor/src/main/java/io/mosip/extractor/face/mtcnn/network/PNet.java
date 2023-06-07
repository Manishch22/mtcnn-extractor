package io.mosip.extractor.face.mtcnn.network;

import java.util.ArrayList;

import org.opencv.core.Mat;

import io.mosip.extractor.constant.MtcnnConstant;
import io.mosip.extractor.face.mtcnn.dto.BoundingBox;
import io.mosip.extractor.face.mtcnn.dto.OrderScore;
import io.mosip.extractor.face.mtcnn.dto.PBox;
import io.mosip.extractor.face.mtcnn.dto.PRelu;
import io.mosip.extractor.face.mtcnn.dto.Weight;
import io.mosip.extractor.face.mtcnn.network.util.NetWorkUtil;
import lombok.Getter;
import lombok.Setter;
import lombok.ToString;

@Getter
@Setter
@ToString
//proposing network
public class PNet extends Network {
	private float nmsThreshold;
	private float pnetThreshold;
	private boolean firstFlag;
	private ArrayList<BoundingBox> boundingBox = new ArrayList<BoundingBox>();
	private ArrayList<OrderScore> bboxScore = new ArrayList<OrderScore>();

	// the image for mxnet conv
	private PBox rgbBox;
	private PBox conv1MatrixBox;
	// the 1th layer's out conv
	private PBox conv1Box;
	private PBox maxPooling1Box;
	private PBox maxPoolingMatrixBox;
	// the 3th layer's out
	private PBox conv2Box;
	private PBox conv3MatrixBox;
	// the 4th layer's out out
	private PBox conv3Box;
	private PBox scoreMatrixBox;
	// the 4th layer's out out
	private PBox scoreBox;
	// the 4th layer's out out
	private PBox locationMatrixBox;
	private PBox locationBox;

	// Weight
	private Weight conv1Weight;
	private PRelu preluGmma1;
	private Weight conv2Weight;
	private PRelu preluGmma2;
	private Weight conv3Weight;
	private PRelu preluGmma3;
	private Weight conv4c1Weight;
	private Weight conv4c2Weight;
	
	public PNet(String fileName) {
		super(fileName);
		setPnetThreshold(MtcnnConstant.PNET_THRESHOLD);
		setNmsThreshold(MtcnnConstant.PNET_NMS_THRESHOLD);
		setFirstFlag(true);
		setRgbBox(new PBox());

		setConv1MatrixBox(new PBox());
		setConv1Box(new PBox());
		setMaxPooling1Box(new PBox());

		setMaxPoolingMatrixBox(new PBox());
		setConv2Box(new PBox());

		setConv3MatrixBox(new PBox());
		setConv3Box(new PBox());

		setScoreMatrixBox(new PBox());
		setScoreBox(new PBox());

		setLocationMatrixBox(new PBox());
		setLocationBox(new PBox());

		setConv1Weight(new Weight());
		setPreluGmma1(new PRelu());
		setConv2Weight(new Weight());
		setPreluGmma2(new PRelu());
		setConv3Weight(new Weight());
		setPreluGmma3(new PRelu());
		setConv4c1Weight(new Weight());
		setConv4c2Weight(new Weight());

		// Init the pnet
		init();
	}

	//Deep cloning
	public PNet(PNet pnet) {
		super(pnet.getFileName());
		setPnetThreshold(pnet.getPnetThreshold());
		setNmsThreshold(pnet.getNmsThreshold());
		setFirstFlag(pnet.isFirstFlag());
		setRgbBox(pnet.getRgbBox());

		setConv1MatrixBox(pnet.getConv1MatrixBox());
		setConv1Box(pnet.getConv1Box());
		setMaxPooling1Box(pnet.getMaxPooling1Box());

		setMaxPoolingMatrixBox(pnet.getMaxPoolingMatrixBox());
		setConv2Box(pnet.getConv2Box());

		setConv3MatrixBox(pnet.getConv3MatrixBox());
		setConv3Box(pnet.getConv3Box());

		setScoreMatrixBox(pnet.getScoreMatrixBox());
		setScoreBox(pnet.getScoreBox());

		setLocationMatrixBox(pnet.getLocationMatrixBox());
		setLocationBox(pnet.getLocationBox());

		setConv1Weight(pnet.getConv1Weight());
		setPreluGmma1(pnet.getPreluGmma1());
		setConv2Weight(pnet.getConv2Weight());
		setPreluGmma2(pnet.getPreluGmma2());
		setConv3Weight(pnet.getConv3Weight());
		setPreluGmma3(pnet.getPreluGmma3());
		setConv4c1Weight(pnet.getConv4c1Weight());
		setConv4c2Weight(pnet.getConv4c2Weight());
	}

	protected void init() {
		// w sc lc ks s p
		long conv1Box = NetWorkUtil.initConvAndFc(getConv1Weight(), 10, 3, 3, 1, 0);
		NetWorkUtil.initpRelu(getPreluGmma1(), 10);
		long conv2Box = NetWorkUtil.initConvAndFc(getConv2Weight(), 16, 10, 3, 1, 0);
		NetWorkUtil.initpRelu(getPreluGmma2(), 16);
		long conv3Box = NetWorkUtil.initConvAndFc(getConv3Weight(), 32, 16, 3, 1, 0);
		NetWorkUtil.initpRelu(getPreluGmma3(), 32);
		long conv4c1 = NetWorkUtil.initConvAndFc(getConv4c1Weight(), 2, 32, 1, 1, 0);
		long conv4c2 = NetWorkUtil.initConvAndFc(getConv4c2Weight(), 4, 32, 1, 1, 0);
		long[] dataNumber = { conv1Box, 10, 10, conv2Box, 16, 16, conv3Box, 32, 32, conv4c1, 2, conv4c2, 4 };

		ArrayList<ArrayList<Float>> pointTeam = new ArrayList<ArrayList<Float>>();
		pointTeam.add(getConv1Weight().getPData());
		pointTeam.add(getConv1Weight().getPBias());
		pointTeam.add(getPreluGmma1().getPData());
		pointTeam.add(getConv2Weight().getPData());
		pointTeam.add(getConv2Weight().getPBias());
		pointTeam.add(getPreluGmma2().getPData());
		pointTeam.add(getConv3Weight().getPData());
		pointTeam.add(getConv3Weight().getPBias());
		pointTeam.add(getPreluGmma3().getPData());
		pointTeam.add(getConv4c1Weight().getPData());
		pointTeam.add(getConv4c1Weight().getPBias());
		pointTeam.add(getConv4c2Weight().getPData());
		pointTeam.add(getConv4c2Weight().getPBias());

		//String filename = "Pnet.txt";
		readNetworkFileData(dataNumber, pointTeam);
	}

	public void run(Mat image, float scale) {
		if (isFirstFlag()) {
			NetWorkUtil.image2MatrixInit(image, getRgbBox());

			NetWorkUtil.feature2MatrixInit(getRgbBox(), getConv1MatrixBox(), getConv1Weight());
			NetWorkUtil.convolutionInit(getConv1Weight(), getRgbBox(), getConv1Box(), getConv1MatrixBox());

			NetWorkUtil.maxPoolingInit(getConv1Box(), getMaxPooling1Box(), 2, 2);
			NetWorkUtil.feature2MatrixInit(getMaxPooling1Box(), getMaxPoolingMatrixBox(), getConv2Weight());
			NetWorkUtil.convolutionInit(getConv2Weight(), getMaxPooling1Box(), getConv2Box(), getMaxPoolingMatrixBox());

			NetWorkUtil.feature2MatrixInit(getConv2Box(), getConv3MatrixBox(), getConv3Weight());
			NetWorkUtil.convolutionInit(getConv3Weight(), getConv2Box(), getConv3Box(), getConv3MatrixBox());

			NetWorkUtil.feature2MatrixInit(getConv3Box(), getScoreMatrixBox(), getConv4c1Weight());
			NetWorkUtil.convolutionInit(getConv4c1Weight(), getConv3Box(), getScoreBox(), getScoreMatrixBox());

			NetWorkUtil.feature2MatrixInit(getConv3Box(), getLocationMatrixBox(), getConv4c2Weight());
			NetWorkUtil.convolutionInit(getConv4c2Weight(), getConv3Box(), getLocationBox(), getLocationMatrixBox());
			setFirstFlag(false);
		}

		NetWorkUtil.image2Matrix(image, getRgbBox());

		NetWorkUtil.feature2Matrix(getRgbBox(), getConv1MatrixBox(), getConv1Weight());
		NetWorkUtil.convolution(getConv1Weight(), getRgbBox(), getConv1Box(), getConv1MatrixBox());
		NetWorkUtil.prelu(getConv1Box(), getConv1Weight().getPBias(), getPreluGmma1().getPData());
		// Pooling layer
		NetWorkUtil.maxPooling(getConv1Box(), getMaxPooling1Box(), 2, 2);

		NetWorkUtil.feature2Matrix(getMaxPooling1Box(), getMaxPoolingMatrixBox(), getConv2Weight());
		NetWorkUtil.convolution(getConv2Weight(), getMaxPooling1Box(), getConv2Box(), getMaxPoolingMatrixBox());
		NetWorkUtil.prelu(getConv2Box(), getConv2Weight().getPBias(), getPreluGmma2().getPData());
		// conv3Box
		NetWorkUtil.feature2Matrix(getConv2Box(), getConv3MatrixBox(), getConv3Weight());
		NetWorkUtil.convolution(getConv3Weight(), getConv2Box(), getConv3Box(), getConv3MatrixBox());
		NetWorkUtil.prelu(getConv3Box(), getConv3Weight().getPBias(), getPreluGmma3().getPData());
		// conv4c1 score
		NetWorkUtil.feature2Matrix(getConv3Box(), getScoreMatrixBox(), getConv4c1Weight());
		NetWorkUtil.convolution(getConv4c1Weight(), getConv3Box(), getScoreBox(), getScoreMatrixBox());
		NetWorkUtil.addBias(getScoreBox(), getConv4c1Weight().getPBias());
		NetWorkUtil.softmax(getScoreBox());

		// conv4c2 location
		NetWorkUtil.feature2Matrix(getConv3Box(), getLocationMatrixBox(), getConv4c2Weight());
		NetWorkUtil.convolution(getConv4c2Weight(), getConv3Box(), getLocationBox(), getLocationMatrixBox());
		NetWorkUtil.addBias(getLocationBox(), getConv4c2Weight().getPBias());
		
		// softmax layer
		generateBoundingBox(getScoreBox(), getLocationBox(), scale);
	}

	private void generateBoundingBox(PBox scoreBox, PBox locationBox, float scale) {
		// for pooling
		int stride = 2;
		int cellSize = 12;
		int count = 0;
		// score p
		int pCount = scoreBox.getWidth() * scoreBox.getHeight();
		int pLocalCount = 0;

		for (int row = 0; row < scoreBox.getHeight(); row++) {
			for (int col = 0; col < scoreBox.getWidth(); col++) {
				if (scoreBox.getPData().get(pCount) > getPnetThreshold()) {
					BoundingBox bbox = new BoundingBox();
					OrderScore order = new OrderScore();

					bbox.setScore (scoreBox.getPData().get(pCount));
					order.setScore (scoreBox.getPData().get(pCount));
					order.setOriOrder(count);
					bbox.setX1 (Math.round((stride * row + 1) / scale));
					bbox.setY1 (Math.round((stride * col + 1) / scale));
					bbox.setX2 (Math.round((stride * row + 1 + cellSize) / scale));
					bbox.setY2 (Math.round((stride * col + 1 + cellSize) / scale));
					bbox.setExist (true);
					bbox.setArea ((bbox.getX2() - bbox.getX1()) * (bbox.getY2() - bbox.getY1()));
					for (int channel = 0; channel < MtcnnConstant.PNET_CHANNELS; channel++)
					{
						bbox.getRegreCoordinates()[channel] = locationBox.getPData()
								.get(pLocalCount + channel * locationBox.getWidth() * locationBox.getHeight());
					}
					getBoundingBox().add(bbox);
					getBboxScore().add(order);
					count++;
				}
				pCount++;
				pLocalCount++;
			}
		}
	}
}