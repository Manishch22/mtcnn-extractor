package io.mosip.extractor.face.mtcnn.network;

import java.util.ArrayList;

import org.opencv.core.Mat;

import io.mosip.extractor.constant.MtcnnConstant;
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
//Refine  network for refining the proposed boundingboxes from predicted from PNet.
public class RNet extends Network {
	private float rnetThreshold;
	private PBox scoreBox;
	private PBox locationBox;

	private PBox rgbBox;

	private PBox conv1MatrixBox;
	private PBox conv1OutBox;
	private PBox pooling1OutBox;

	private PBox conv2MatrixBox;
	private PBox conv2OutBox;
	private PBox pooling2OutBox;

	private PBox conv3MatrixBox;
	private PBox conv3OutBox;

	private PBox fc4OutBox;

	// Weight
	private Weight conv1Weight;
	private PRelu preluGmma1;
	private Weight conv2Weight;
	private PRelu preluGmma2;
	private Weight conv3Weight;
	private PRelu preluGmma3;
	private Weight fc4Weight;
	private PRelu preluGmma4;
	private Weight scoreWeight;
	private Weight locationWeight;

	public RNet(String fileName) {
		super(fileName);
		setRnetThreshold(MtcnnConstant.RNET_THRESHOLD);
		setScoreBox(new PBox());
		setLocationBox(new PBox());

		setRgbBox(new PBox());
		setConv1MatrixBox (new PBox());
		setConv1OutBox (new PBox());
		setPooling1OutBox (new PBox());

		setConv2MatrixBox (new PBox());
		setConv2OutBox (new PBox());
		setPooling2OutBox (new PBox());

		setConv3MatrixBox (new PBox());
		setConv3OutBox (new PBox());

		setFc4OutBox (new PBox());

		setConv1Weight (new Weight());
		setPreluGmma1 (new PRelu());
		setConv2Weight (new Weight());
		setPreluGmma2 (new PRelu());
		setConv3Weight (new Weight());
		setPreluGmma3 (new PRelu());
		setFc4Weight (new Weight());
		setPreluGmma4 (new PRelu());
		setScoreWeight (new Weight());
		setLocationWeight (new Weight());
		// Init the rnet
		init();
	}

	//Deep copy
	public RNet(RNet rnet) {
		super(rnet.getFileName());
		
		setRnetThreshold(rnet.getRnetThreshold());
		setScoreBox(rnet.getScoreBox());
		setLocationBox(rnet.getLocationBox());

		setRgbBox(rnet.getRgbBox());
		setConv1MatrixBox (rnet.getConv1MatrixBox());
		setConv1OutBox (rnet.getConv1OutBox());
		setPooling1OutBox (rnet.getPooling1OutBox());

		setConv2MatrixBox (rnet.getConv2MatrixBox());
		setConv2OutBox (rnet.getConv2OutBox());
		setPooling2OutBox (rnet.getPooling2OutBox());

		setConv3MatrixBox (rnet.getConv3MatrixBox());
		setConv3OutBox (rnet.getConv3OutBox());

		setFc4OutBox (rnet.getFc4OutBox());

		setConv1Weight (rnet.getConv1Weight());
		setPreluGmma1 (rnet.getPreluGmma1());
		setConv2Weight (rnet.getConv2Weight());
		setPreluGmma2 (rnet.getPreluGmma2());
		setConv3Weight (rnet.getConv3Weight());
		setPreluGmma3 (rnet.getPreluGmma3());
		setFc4Weight (rnet.getFc4Weight());
		setPreluGmma4 (rnet.getPreluGmma4());
		setScoreWeight (rnet.getScoreWeight());
		setLocationWeight (rnet.getLocationWeight());
	}
	
	protected void init() {
		// // w sc lc ks s p
		long conv1 = NetWorkUtil.initConvAndFc(getConv1Weight(), 28, 3, 3, 1, 0);
		NetWorkUtil.initpRelu(getPreluGmma1(), 28);
		long conv2 = NetWorkUtil.initConvAndFc(getConv2Weight(), 48, 28, 3, 1, 0);
		NetWorkUtil.initpRelu(getPreluGmma2(), 48);
		long conv3 = NetWorkUtil.initConvAndFc(getConv3Weight(), 64, 48, 2, 1, 0);
		NetWorkUtil.initpRelu(getPreluGmma3(), 64);
		long fc4 = NetWorkUtil.initConvAndFc(getFc4Weight(), 128, 576, 1, 1, 0);
		NetWorkUtil.initpRelu(getPreluGmma4(), 128);
		long score = NetWorkUtil.initConvAndFc(getScoreWeight(), 2, 128, 1, 1, 0);
		long location = NetWorkUtil.initConvAndFc(getLocationWeight(), 4, 128, 1, 1, 0);
		long[] dataNumber = { conv1, 28, 28, conv2, 48, 48, conv3, 64, 64, fc4, 128, 128, score, 2, location, 4 };

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
		pointTeam.add(getFc4Weight().getPData());
		pointTeam.add(getFc4Weight().getPBias());
		pointTeam.add(getPreluGmma4().getPData());
		pointTeam.add(getScoreWeight().getPData());
		pointTeam.add(getScoreWeight().getPBias());
		pointTeam.add(getLocationWeight().getPData());
		pointTeam.add(getLocationWeight().getPBias());

		//String filename = "Rnet.txt";
		readNetworkFileData(dataNumber, pointTeam);
		
		image2MatrixInit(getRgbBox(), MtcnnConstant.RGB_CHANNELS, MtcnnConstant.RNET_IMAGE_MATRIX_WIDTH, MtcnnConstant.RNET_IMAGE_MATRIX_HEIGHT);
		NetWorkUtil.feature2MatrixInit(getRgbBox(), getConv1MatrixBox(), getConv1Weight());
		NetWorkUtil.convolutionInit(getConv1Weight(), getRgbBox(), getConv1OutBox(), getConv1MatrixBox());
		NetWorkUtil.maxPoolingInit(getConv1OutBox(), getPooling1OutBox(), 3, 2);
		NetWorkUtil.feature2MatrixInit(getPooling1OutBox(), getConv2MatrixBox(), getConv2Weight());
		NetWorkUtil.convolutionInit(getConv2Weight(), getPooling1OutBox(), getConv2OutBox(), getConv2MatrixBox());
		NetWorkUtil.maxPoolingInit(getConv2OutBox(), getPooling2OutBox(), 3, 2);
		NetWorkUtil.feature2MatrixInit(getPooling2OutBox(), getConv3MatrixBox(), getConv3Weight());
		NetWorkUtil.convolutionInit(getConv3Weight(), getPooling2OutBox(), getConv3OutBox(), getConv3MatrixBox());
		NetWorkUtil.fullConnectInit(getFc4Weight(), getFc4OutBox());
		NetWorkUtil.fullConnectInit(getScoreWeight(), getScoreBox());
		NetWorkUtil.fullConnectInit(getLocationWeight(), getLocationBox());
	}
	
	public void run(Mat image) {
		NetWorkUtil.image2Matrix(image, getRgbBox());

		NetWorkUtil.feature2Matrix(getRgbBox(), getConv1MatrixBox(), getConv1Weight());
		NetWorkUtil.convolution(getConv1Weight(), getRgbBox(), getConv1OutBox(), getConv1MatrixBox());
		NetWorkUtil.prelu(getConv1OutBox(), getConv1Weight().getPBias(), getPreluGmma1().getPData());

		NetWorkUtil.maxPooling(getConv1OutBox(), getPooling1OutBox(), 3, 2);

		NetWorkUtil.feature2Matrix(getPooling1OutBox(), getConv2MatrixBox(), getConv2Weight());
		NetWorkUtil.convolution(getConv2Weight(), getPooling1OutBox(), getConv2OutBox(), getConv2MatrixBox());
		NetWorkUtil.prelu(getConv2OutBox(), getConv2Weight().getPBias(), getPreluGmma2().getPData());
		NetWorkUtil.maxPooling(getConv2OutBox(), getPooling2OutBox(), 3, 2);

		// conv3
		NetWorkUtil.feature2Matrix(getPooling2OutBox(), getConv3MatrixBox(), getConv3Weight());
		NetWorkUtil.convolution(getConv3Weight(), getPooling2OutBox(), getConv3OutBox(), getConv3MatrixBox());
		NetWorkUtil.prelu(getConv3OutBox(), getConv3Weight().getPBias(), getPreluGmma3().getPData());

		// flatten
		NetWorkUtil.fullConnect(getFc4Weight(), getConv3OutBox(), getFc4OutBox());
		NetWorkUtil.prelu(getFc4OutBox(), getFc4Weight().getPBias(), getPreluGmma4().getPData());

		// conv51 score
		NetWorkUtil.fullConnect(getScoreWeight(), getFc4OutBox(), getScoreBox());
		NetWorkUtil.addBias(getScoreBox(), getScoreWeight().getPBias());
		NetWorkUtil.softmax(getScoreBox());

		// conv5_2 location
		NetWorkUtil.fullConnect(getLocationWeight(), getFc4OutBox(), getLocationBox());
		NetWorkUtil.addBias(getLocationBox(), getLocationWeight().getPBias());
		// pBoxShow(location_);
	}
}
