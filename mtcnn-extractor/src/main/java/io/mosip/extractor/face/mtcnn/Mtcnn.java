package io.mosip.extractor.face.mtcnn;

import java.util.ArrayList;

import org.opencv.core.Mat;
import org.opencv.core.MatOfByte;
import org.opencv.core.Rect;
import org.opencv.core.Size;
import org.opencv.imgcodecs.Imgcodecs;
import org.opencv.imgproc.Imgproc;

import io.mosip.biometrics.util.ConvertRequestDto;
import io.mosip.biometrics.util.face.FaceBDIR;
import io.mosip.biometrics.util.face.FaceDecoder;
import io.mosip.extractor.constant.ExtractorErrorCode;
import io.mosip.extractor.constant.MtcnnConstant;
import io.mosip.extractor.exception.ExtractorException;
import io.mosip.extractor.face.mtcnn.dto.BoundingBox;
import io.mosip.extractor.face.mtcnn.dto.FaceInfo;
import io.mosip.extractor.face.mtcnn.dto.OrderScore;
import io.mosip.extractor.face.mtcnn.network.ONet;
import io.mosip.extractor.face.mtcnn.network.PNet;
import io.mosip.extractor.face.mtcnn.network.RNet;
import io.mosip.extractor.face.mtcnn.network.util.NetWorkUtil;
import io.mosip.extractor.face.mtcnn.util.Util;
import lombok.Getter;
import lombok.Setter;
import lombok.ToString;

@Getter
@Setter
@ToString
//Multi-task Cascaded Convolutional Network (MTCNN) model for face detection.
public class Mtcnn {
	private Mat resizeImage = new Mat();
	private float[] nmsThreshold = {MtcnnConstant.NMS_THRESHOLD, MtcnnConstant.NMS_THRESHOLD, MtcnnConstant.NMS_THRESHOLD };
	private ArrayList<Float> scales = null;

	private ArrayList<PNet> simpleFacePnet = null;
	private ArrayList<BoundingBox> firstBbox = null;
	private ArrayList<OrderScore> firstOrderScore = null;

	private RNet refineNet = null;
	private RNet refineNetClone = null;
	private ArrayList<BoundingBox> secondBbox = null;
	private ArrayList<OrderScore> secondBboxScore = null;

	private ONet outNet = null;
	private ONet outNetClone = null;
	private ArrayList<BoundingBox> thirdBbox = null;
	private ArrayList<OrderScore> thirdBboxScore = null;

	private PNet pNet = null;
	private PNet pNetClone = null;
	private boolean isInit = false;
	
	public Mtcnn() {
		super();
		initMtcnnNetwork();
	}
	
	private void initMtcnnNetwork() {		
		if (!this.isInit())
		{
			setPNet(new PNet("Pnet.txt"));
			setPNetClone(new PNet(getPNet()));
			setRefineNet(new RNet("Rnet.txt"));
			setRefineNetClone(new RNet(getRefineNet()));
			setOutNet(new ONet("Onet.txt"));
			setOutNetClone(new ONet(getOutNet()));

			setScales(new ArrayList<Float>());

			setSimpleFacePnet (new ArrayList<PNet>());
			setFirstBbox (new ArrayList<BoundingBox>());
			setFirstOrderScore (new ArrayList<OrderScore>());

			setSecondBbox (new ArrayList<BoundingBox>());
			setSecondBboxScore (new ArrayList<OrderScore>());

			setThirdBbox (new ArrayList<BoundingBox>());
			setThirdBboxScore (new ArrayList<OrderScore>());
			setInit(true);
		}
		else
		{
			setPNet(new PNet(getPNetClone()));
			setRefineNet(new RNet(getRefineNetClone()));
			setOutNet(new ONet(getOutNetClone()));
			
			getScales().clear();
			getSimpleFacePnet().clear();
			// inital bbox
			getFirstBbox().clear();
			getFirstOrderScore().clear();
			getSecondBbox().clear();
			getSecondBboxScore().clear();
			getThirdBbox().clear();
			getThirdBboxScore().clear();
		}		
	}
	
	private void initMtcnnForFaceImage(int rows, int cols) {
		initMtcnnNetwork();
		float minl = rows > cols ? cols : rows;
		int MIN_DET_SIZE = 12;
		int minsize = 60;
		float m = (float) MIN_DET_SIZE / minsize;
		minl *= m;
		float factor = 0.709F;
		int factor_count = 0;

		while (minl > MIN_DET_SIZE) {
			if (factor_count > 0) {
				m = m * factor;
			}
			getScales().add(m);
			minl *= factor;
			factor_count++;
		}
		for (int i = 0; i < getScales().size(); i++) {
			getSimpleFacePnet().add(new PNet(getPNet()));
		}
	}

	public ArrayList<FaceInfo> findFace(byte[] imageData) {
		Mat image = Imgcodecs.imdecode(new MatOfByte(imageData), Imgcodecs.IMREAD_UNCHANGED);//Imgcodecs.imread(fileName);
		return findFace(image);
	}
	
	public ArrayList<FaceInfo> findFace(String imageISOData) {
		return findFace(getFaceDataFromISO (imageISOData));
	}

	public ArrayList<FaceInfo> findFace(Mat image) {
		// inital Mtcnn For Face Image
		initMtcnnForFaceImage(image.rows(), image.cols());
		// inital bbox
		getFirstBbox().clear();
		getFirstOrderScore().clear();
		getSecondBbox().clear();
		getSecondBboxScore().clear();
		getThirdBbox().clear();
		getThirdBboxScore().clear();
		
		// initial output face
		ArrayList<FaceInfo> faces = new ArrayList<FaceInfo>();

		int count = 0;
		for (int i = 0; i < getScales().size(); i++) {
			int changedH = (int) Math.ceil(image.rows() * getScales().get(i));
			int changedW = (int) Math.ceil(image.cols() * getScales().get(i));
			Imgproc.resize(image, this.getResizeImage(), new Size(changedW, changedH));
			getSimpleFacePnet().get(i).run(this.getResizeImage(), getScales().get(i));
			NetWorkUtil.nms(getSimpleFacePnet().get(i).getBoundingBox(), getSimpleFacePnet().get(i).getBboxScore(),
					getSimpleFacePnet().get(i).getNmsThreshold(), MtcnnConstant.BOUNDINGBOX_MODEL_UNION);
			for (int k = 0; k < getSimpleFacePnet().get(i).getBoundingBox().size(); k++) {
				if (getSimpleFacePnet().get(i).getBoundingBox().get(k).isExist()) {
					getFirstBbox().add(getSimpleFacePnet().get(i).getBoundingBox().get(k));
					OrderScore order = new OrderScore();
					order.setScore (getSimpleFacePnet().get(i).getBoundingBox().get(k).getScore());
					order.setOriOrder(count);
					getFirstOrderScore().add(order);
					count++;
				}
			}
			getSimpleFacePnet().get(i).getBboxScore().clear();
			getSimpleFacePnet().get(i).getBoundingBox().clear();
		}
		// the first stage's nms
		if (count < 1) {
			return faces;
		}
		
		NetWorkUtil.nms(getFirstBbox(), getFirstOrderScore(), getNmsThreshold()[0], MtcnnConstant.BOUNDINGBOX_MODEL_UNION);
		NetWorkUtil.refineAndSquareBbox(getFirstBbox(), image.rows(), image.cols());

		// second stage
		count = 0;
		for (int i = 0; i < getFirstBbox().size(); i++) {
			if (getFirstBbox().get(i).isExist()) {
				Rect temp = new Rect(getFirstBbox().get(i).getY1(), getFirstBbox().get(i).getX1(),
						getFirstBbox().get(i).getY2() - getFirstBbox().get(i).getY1(), getFirstBbox().get(i).getX2() - getFirstBbox().get(i).getX1());
				Mat secImage = new Mat();
				Mat rectImage = new Mat(image, temp);
				Imgproc.resize(rectImage, secImage, new Size(24, 24));
				getRefineNet().run(secImage);
				if (getRefineNet().getScoreBox().getPData().get(1) > getRefineNet().getRnetThreshold()) {
					for (int k = 0; k < 4; k++) {
						getFirstBbox().get(i).getRegreCoordinates()[k] = getRefineNet().getLocationBox().getPData().get(k);
					}
					getFirstBbox().get(i).setArea ((getFirstBbox().get(i).getX2() - getFirstBbox().get(i).getX1())
							* (getFirstBbox().get(i).getY2() - getFirstBbox().get(i).getY1()));
					getFirstBbox().get(i).setScore (getRefineNet().getScoreBox().getPData().get(1));
					getSecondBbox().add(getFirstBbox().get(i));
					OrderScore order = new OrderScore();
					order.setScore (getFirstBbox().get(i).getScore());
					order.setOriOrder(count++);
					getSecondBboxScore().add(order);
				} else {
					getFirstBbox().get(i).setExist (false);
				}
			}
		}
		if (count < 1) {
			return faces;
		}
		
		NetWorkUtil.nms(getSecondBbox(), getSecondBboxScore(), getNmsThreshold()[1], MtcnnConstant.BOUNDINGBOX_MODEL_UNION);
		NetWorkUtil.refineAndSquareBbox(getSecondBbox(), image.rows(), image.cols());

		// third stage
		count = 0;
		for (int i = 0; i < getSecondBbox().size(); i++) {
			if (getSecondBbox().get(i).isExist()) {
				Rect temp = new Rect(getSecondBbox().get(i).getY1(), getSecondBbox().get(i).getX1(),
						getSecondBbox().get(i).getY2() - getSecondBbox().get(i).getY1(), getSecondBbox().get(i).getX2() - getSecondBbox().get(i).getX1());
				Mat thirdImage = new Mat();
				Mat rectImage = new Mat(image, temp);
				Imgproc.resize(rectImage, thirdImage, new Size(48, 48));
				getOutNet().run(thirdImage);
				ArrayList<Float> pp = null;
				if (getOutNet().getScoreBox().getPData().get(1) > getOutNet().getOnetThreshold()) {
					for (int k = 0; k < 4; k++) {
						getSecondBbox().get(i).getRegreCoordinates()[k] = getOutNet().getLocationBox().getPData().get(k);
					}
					getSecondBbox().get(i).setArea ((getSecondBbox().get(i).getX2() - getSecondBbox().get(i).getX1())
							* (getSecondBbox().get(i).getY2() - getSecondBbox().get(i).getY1()));
					getSecondBbox().get(i).setScore (getOutNet().getScoreBox().getPData().get(1));

					pp = getOutNet().getKeyPointBox().getPData();
					for (int num = 0; num < 5; num++) {
						getSecondBbox().get(i).getKeyPoints()[num] = getSecondBbox().get(i).getY1()
								+ (getSecondBbox().get(i).getY2() - getSecondBbox().get(i).getY1()) * (pp.get(num));
					}
					for (int num = 0; num < 5; num++) {
						getSecondBbox().get(i).getKeyPoints()[num + 5] = getSecondBbox().get(i).getX1()
								+ (getSecondBbox().get(i).getX2() - getSecondBbox().get(i).getX1()) * (pp.get(num + 5));
					}

					getThirdBbox().add(getSecondBbox().get(i));
					OrderScore order = new OrderScore();
					order.setScore (getSecondBbox().get(i).getScore());
					order.setOriOrder(count++);
					getThirdBboxScore().add(order);
				} else {
					getSecondBbox().get(i).setExist (false);
				}
			}
		}
		if (count < 1) {
			return faces;
		}
		
		NetWorkUtil.refineAndSquareBbox(getThirdBbox(), image.rows(), image.cols());
		NetWorkUtil.nms(getThirdBbox(), getThirdBboxScore(), getNmsThreshold()[2], MtcnnConstant.BOUNDINGBOX_MODEL_MINIMUM);

		for (int i = 0; i < getThirdBbox().size(); i++) {
			if (getThirdBbox().get(i).isExist()) {
				FaceInfo faceInfo = new FaceInfo();

				// face rect
				faceInfo.setFaceRect(new Rect(getThirdBbox().get(i).getY1(), getThirdBbox().get(i).getX1(),
						getThirdBbox().get(i).getY2() - getThirdBbox().get(i).getY1() + 1,
						getThirdBbox().get(i).getX2() - getThirdBbox().get(i).getX1() + 1));

				// face keypoint
				faceInfo.setKeyPoints(new ArrayList<Float>());
				for (int num = 0; num < MtcnnConstant.KEY_POINTS; num++) {
					faceInfo.getKeyPoints().add(getThirdBbox().get(i).getKeyPoints()[num]);
				}

				faces.add(faceInfo);
			}
		}
		return faces;
	}
	
	public byte[] getFaceDataFromISO(String imageISOData)
	{
		ExtractorErrorCode errorCode = null;
		if (imageISOData == null || imageISOData.length() == 0)
		{
			errorCode = ExtractorErrorCode.SOURCE_CAN_NOT_BE_EMPTY_OR_NULL_EXCEPTION;
			throw new ExtractorException (errorCode.getErrorCode(), errorCode.getErrorMessage());
		}
		byte [] imageData = getFaceImageData(imageISOData);				
		if (imageData == null || imageData.length == 0)
		{
			errorCode = ExtractorErrorCode.COULD_NOT_READ_ISO_IMAGE_DATA_EXCEPTION;
			throw new ExtractorException (errorCode.getErrorCode(), errorCode.getErrorMessage());
		}
		
		return imageData;
	}
	
	private byte[] getFaceImageData(String isoData) {
		ExtractorErrorCode errorCode = null;
		ConvertRequestDto requestDto = new ConvertRequestDto();
		requestDto.setModality("Face");
		requestDto.setVersion("ISO19794_5_2011");
		try {
			requestDto.setInputBytes(Util.decodeURLSafeBase64 (isoData));
		} catch (Exception e) {
			errorCode = ExtractorErrorCode.SOURCE_NOT_VALID_BASE64URLENCODED_EXCEPTION;
			throw new ExtractorException (errorCode.getErrorCode(), e.getLocalizedMessage());
		}

		FaceBDIR bdir;
		try {
			bdir = FaceDecoder.getFaceBDIR(requestDto);
		} catch (Exception e) {
			errorCode = ExtractorErrorCode.SOURCE_NOT_VALID_FACE_ISO_FORMAT_EXCEPTION;
			throw new ExtractorException (errorCode.getErrorCode(), e.getLocalizedMessage());
		}

		return bdir.getImage();
	}
}