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
// output network uses both PNet & RNet before taking the output of RNet as input for final prediction.
public class ONet extends Network {
	private float onetThreshold;
	private PBox scoreBox;
	private PBox locationBox;
	private PBox keyPointBox;

	private PBox rgbBox;
	private PBox conv1MatrixBox;
	private PBox conv1OutBox;
	private PBox pooling1OutBox;

	private PBox conv2MatrixBox;
	private PBox conv2OutBox;
	private PBox pooling2OutBox;

	private PBox conv3MatrixBox;
	private PBox conv3OutBox;
	private PBox pooling3OutBox;

	private PBox conv4MatrixBox;
	private PBox conv4OutBox;

	private PBox fc5OutBox;

	// Weight
	private Weight conv1Weight;
	private PRelu preluGmma1;
	private Weight conv2Weight;
	private PRelu preluGmma2;
	private Weight conv3Weight;
	private PRelu preluGmma3;
	private Weight conv4Weight;
	private PRelu preluGmma4;
	private Weight fc5Weight;
	private PRelu preluGmma5;
	private Weight scoreWeight;
	private Weight locationWeight;
	private Weight keyPointWeight;
	
	public ONet(String fileName) {
		super(fileName);
		setOnetThreshold(MtcnnConstant.ONET_THRESHOLD);
		setRgbBox(new PBox());

		setConv1MatrixBox(new PBox());
		setConv1OutBox(new PBox());
		setPooling1OutBox(new PBox());

		setConv2MatrixBox(new PBox());
		setConv2OutBox(new PBox());
		setPooling2OutBox(new PBox());

		setConv3MatrixBox(new PBox());
		setConv3OutBox(new PBox());
		setPooling3OutBox(new PBox());

		setConv4MatrixBox(new PBox());
		setConv4OutBox(new PBox());

		setFc5OutBox(new PBox());

		setScoreBox(new PBox());
		setLocationBox(new PBox());
		setKeyPointBox(new PBox());

		setConv1Weight(new Weight());
		setPreluGmma1(new PRelu());
		setConv2Weight(new Weight());
		setPreluGmma2(new PRelu());
		setConv3Weight(new Weight());
		setPreluGmma3(new PRelu());
		setConv4Weight(new Weight());
		setPreluGmma4(new PRelu());
		setFc5Weight(new Weight());
		setPreluGmma5(new PRelu());
		setScoreWeight(new Weight());
		setLocationWeight(new Weight());
		setKeyPointWeight(new Weight());
		// Init the Onet
		init();	
	}
	
	//Deep cloning
	public ONet(ONet onet) {
		super(onet.getFileName());

		setOnetThreshold(onet.getOnetThreshold());
		setRgbBox(onet.getRgbBox());

		setConv1MatrixBox(onet.getConv1MatrixBox());
		setConv1OutBox(onet.getConv1OutBox());
		setPooling1OutBox(onet.getPooling1OutBox());

		setConv2MatrixBox(onet.getConv2MatrixBox());
		setConv2OutBox(onet.getConv2OutBox());
		setPooling2OutBox(onet.getPooling2OutBox());

		setConv3MatrixBox(onet.getConv3MatrixBox());
		setConv3OutBox(onet.getConv3OutBox());
		setPooling3OutBox(onet.getPooling3OutBox());

		setConv4MatrixBox(onet.getConv4MatrixBox());
		setConv4OutBox(onet.getConv4OutBox());

		setFc5OutBox(onet.getFc5OutBox());

		setScoreBox(onet.getScoreBox());
		setLocationBox(onet.getLocationBox());
		setKeyPointBox(onet.getKeyPointBox());

		setConv1Weight(onet.getConv1Weight());
		setPreluGmma1(onet.getPreluGmma1());
		setConv2Weight(onet.getConv2Weight());
		setPreluGmma2(onet.getPreluGmma2());
		setConv3Weight(onet.getConv3Weight());
		setPreluGmma3(onet.getPreluGmma3());
		setConv4Weight(onet.getConv4Weight());
		setPreluGmma4(onet.getPreluGmma4());
		setFc5Weight(onet.getFc5Weight());
		setPreluGmma5(onet.getPreluGmma5());
		setScoreWeight(onet.getScoreWeight());
		setLocationWeight(onet.getLocationWeight());
		setKeyPointWeight(onet.getKeyPointWeight());		
	}
	
	protected void init() {
		// // w sc lc ks s p
		long conv1 = NetWorkUtil.initConvAndFc(getConv1Weight(), 32, 3, 3, 1, 0);
		NetWorkUtil.initpRelu(getPreluGmma1(), 32);
		long conv2 = NetWorkUtil.initConvAndFc(getConv2Weight(), 64, 32, 3, 1, 0);
		NetWorkUtil.initpRelu(getPreluGmma2(), 64);
		long conv3 = NetWorkUtil.initConvAndFc(getConv3Weight(), 64, 64, 3, 1, 0);
		NetWorkUtil.initpRelu(getPreluGmma3(), 64);
		long conv4 = NetWorkUtil.initConvAndFc(getConv4Weight(), 128, 64, 2, 1, 0);
		NetWorkUtil.initpRelu(getPreluGmma4(), 128);
		long fc5 = NetWorkUtil.initConvAndFc(getFc5Weight(), 256, 1152, 1, 1, 0);
		NetWorkUtil.initpRelu(getPreluGmma5(), 256);
		long score = NetWorkUtil.initConvAndFc(getScoreWeight(), 2, 256, 1, 1, 0);
		long location = NetWorkUtil.initConvAndFc(getLocationWeight(), 4, 256, 1, 1, 0);
		long keyPoint = NetWorkUtil.initConvAndFc(getKeyPointWeight(), 10, 256, 1, 1, 0);
		long[] dataNumber = { conv1, 32, 32, conv2, 64, 64, conv3, 64, 64, conv4, 128, 128, fc5, 256, 256, score, 2,
				location, 4, keyPoint, 10 };

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
		pointTeam.add(getConv4Weight().getPData());
		pointTeam.add(getConv4Weight().getPBias());
		pointTeam.add(getPreluGmma4().getPData());
		pointTeam.add(getFc5Weight().getPData());
		pointTeam.add(getFc5Weight().getPBias());
		pointTeam.add(getPreluGmma5().getPData());
		pointTeam.add(getScoreWeight().getPData());
		pointTeam.add(getScoreWeight().getPBias());
		pointTeam.add(getLocationWeight().getPData());
		pointTeam.add(getLocationWeight().getPBias());
		pointTeam.add(getKeyPointWeight().getPData());
		pointTeam.add(getKeyPointWeight().getPBias());

		//String filename = "Onet.txt";
		readNetworkFileData(dataNumber, pointTeam);

		// Init the network
		image2MatrixInit(getRgbBox(), MtcnnConstant.RGB_CHANNELS, MtcnnConstant.ONET_IMAGE_MATRIX_WIDTH, MtcnnConstant.ONET_IMAGE_MATRIX_HEIGHT);

		NetWorkUtil.feature2MatrixInit(getRgbBox(), getConv1MatrixBox(), getConv1Weight());
		NetWorkUtil.convolutionInit(getConv1Weight(), getRgbBox(), getConv1OutBox(), getConv1MatrixBox());
		NetWorkUtil.maxPoolingInit(getConv1OutBox(), getPooling1OutBox(), 3, 2);

		NetWorkUtil.feature2MatrixInit(getPooling1OutBox(), getConv2MatrixBox(), getConv2Weight());
		NetWorkUtil.convolutionInit(getConv2Weight(), getPooling1OutBox(), getConv2OutBox(), getConv2MatrixBox());
		NetWorkUtil.maxPoolingInit(getConv2OutBox(), getPooling2OutBox(), 3, 2);

		NetWorkUtil.feature2MatrixInit(getPooling2OutBox(), getConv3MatrixBox(), getConv3Weight());
		NetWorkUtil.convolutionInit(getConv3Weight(), getPooling2OutBox(), getConv3OutBox(), getConv3MatrixBox());
		NetWorkUtil.maxPoolingInit(getConv3OutBox(), getPooling3OutBox(), 2, 2);

		NetWorkUtil.feature2MatrixInit(getPooling3OutBox(), getConv4MatrixBox(), getConv4Weight());
		NetWorkUtil.convolutionInit(getConv4Weight(), getPooling3OutBox(), getConv4OutBox(), getConv4MatrixBox());

		NetWorkUtil.fullConnectInit(getFc5Weight(), getFc5OutBox());
		NetWorkUtil.fullConnectInit(getScoreWeight(), getScoreBox());
		NetWorkUtil.fullConnectInit(getLocationWeight(), getLocationBox());
		NetWorkUtil.fullConnectInit(getKeyPointWeight(), getKeyPointBox());	
	}

	public void run(Mat image) {
		NetWorkUtil.image2Matrix(image, getRgbBox());

		// conv1
		NetWorkUtil.feature2Matrix(getRgbBox(), getConv1MatrixBox(), getConv1Weight());
		NetWorkUtil.convolution(getConv1Weight(), getRgbBox(), getConv1OutBox(), getConv1MatrixBox());
		NetWorkUtil.prelu(getConv1OutBox(), getConv1Weight().getPBias(), getPreluGmma1().getPData());
		NetWorkUtil.maxPooling(getConv1OutBox(), getPooling1OutBox(), 3, 2);

		// conv2
		NetWorkUtil.feature2Matrix(getPooling1OutBox(), getConv2MatrixBox(), getConv2Weight());
		NetWorkUtil.convolution(getConv2Weight(), getPooling1OutBox(), getConv2OutBox(), getConv2MatrixBox());
		NetWorkUtil.prelu(getConv2OutBox(), getConv2Weight().getPBias(), getPreluGmma2().getPData());
		NetWorkUtil.maxPooling(getConv2OutBox(), getPooling2OutBox(), 3, 2);

		// conv3
		NetWorkUtil.feature2Matrix(getPooling2OutBox(), getConv3MatrixBox(), getConv3Weight());
		NetWorkUtil.convolution(getConv3Weight(), getPooling2OutBox(), getConv3OutBox(), getConv3MatrixBox());
		NetWorkUtil.prelu(getConv3OutBox(), getConv3Weight().getPBias(), getPreluGmma3().getPData());
		NetWorkUtil.maxPooling(getConv3OutBox(), getPooling3OutBox(), 2, 2);

		// conv4
		NetWorkUtil.feature2Matrix(getPooling3OutBox(), getConv4MatrixBox(), getConv4Weight());
		NetWorkUtil.convolution(getConv4Weight(), getPooling3OutBox(), getConv4OutBox(), getConv4MatrixBox());
		NetWorkUtil.prelu(getConv4OutBox(), getConv4Weight().getPBias(), getPreluGmma4().getPData());

		NetWorkUtil.fullConnect(getFc5Weight(), getConv4OutBox(), getFc5OutBox());
		NetWorkUtil.prelu(getFc5OutBox(), getFc5Weight().getPBias(), getPreluGmma5().getPData());

		// conv6_1 score
		NetWorkUtil.fullConnect(getScoreWeight(), getFc5OutBox(), getScoreBox());
		NetWorkUtil.addBias(getScoreBox(), getScoreWeight().getPBias());
		NetWorkUtil.softmax(getScoreBox());

		// conv6_2 location
		NetWorkUtil.fullConnect(getLocationWeight(), getFc5OutBox(), getLocationBox());
		NetWorkUtil.addBias(getLocationBox(), getLocationWeight().getPBias());

		// conv6_2 location
		NetWorkUtil.fullConnect(getKeyPointWeight(), getFc5OutBox(), getKeyPointBox());
		NetWorkUtil.addBias(getKeyPointBox(), getKeyPointWeight().getPBias());
	}
}
