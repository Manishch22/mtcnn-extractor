package io.mosip.extractor.face.sdk.impl;

import java.util.ArrayList;
import java.util.Collections;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

import org.opencv.core.Mat;
import org.opencv.core.MatOfByte;
import org.opencv.core.MatOfInt;
import org.opencv.core.Rect;
import org.opencv.core.Size;
import org.opencv.imgcodecs.Imgcodecs;
import org.opencv.imgproc.Imgproc;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.boot.autoconfigure.EnableAutoConfiguration;
import org.springframework.stereotype.Component;

import io.mosip.extractor.constant.ExtractorErrorCode;
import io.mosip.extractor.exception.ExtractorException;
import io.mosip.extractor.face.mtcnn.Mtcnn;
import io.mosip.extractor.face.mtcnn.dto.FaceInfo;
import io.mosip.extractor.face.mtcnn.dto.FaceRect;
import io.mosip.extractor.face.mtcnn.dto.FaceRectComparator;
import io.mosip.extractor.face.mtcnn.util.Util;
import io.mosip.kernel.biometrics.constant.BiometricFunction;
import io.mosip.kernel.biometrics.constant.BiometricType;
import io.mosip.kernel.biometrics.constant.ProcessedLevelType;
import io.mosip.kernel.biometrics.entities.BIR;
import io.mosip.kernel.biometrics.entities.BiometricRecord;
import io.mosip.kernel.biometrics.model.MatchDecision;
import io.mosip.kernel.biometrics.model.QualityCheck;
import io.mosip.kernel.biometrics.model.Response;
import io.mosip.kernel.biometrics.model.SDKInfo;
import io.mosip.kernel.biometrics.spi.IBioApiV2;

/**
 * The Class BioApiImpl.
 * 
 * @author Janardhan B S
 * 
 */
@Component
@EnableAutoConfiguration
public class FaceSDKImpl implements IBioApiV2 {
	Logger LOGGER = LoggerFactory.getLogger(FaceSDKImpl.class);
	private Mtcnn mtcnn = new Mtcnn();
	
	private static final String API_VERSION = "0.9";
	@Override
	public SDKInfo init(Map<String, String> initParams) {
		// TODO validate for mandatory initParams
		SDKInfo sdkInfo = new SDKInfo(API_VERSION, "sample", "sample", "sample");
		List<BiometricType> supportedModalities = new ArrayList<>();
		supportedModalities.add(BiometricType.FACE);
		sdkInfo.setSupportedModalities(supportedModalities);
		Map<BiometricFunction, List<BiometricType>> supportedMethods = new HashMap<>();
		supportedMethods.put(BiometricFunction.EXTRACT, supportedModalities);
		sdkInfo.setSupportedMethods(supportedMethods);
		return sdkInfo;	
	}

	@Override
	public Response<QualityCheck> checkQuality(BiometricRecord sample, List<BiometricType> modalitiesToCheck,
			Map<String, String> flags) {
		// TODO Auto-generated method stub
		return null;
	}

	@Override
	public Response<MatchDecision[]> match(BiometricRecord sample, BiometricRecord[] gallery,
			List<BiometricType> modalitiesToMatch, Map<String, String> flags) {
		// TODO Auto-generated method stub
		return null;
	}

	// Face extract only
	@Override
	public Response<BiometricRecord> extractTemplate(BiometricRecord record, List<BiometricType> modalitiesToExtract,
			Map<String, String> flags) {
		Response<BiometricRecord> response = new Response<>();
		Map<String, String> values = new HashMap<>();
		for (BIR segment : record.getSegments()) {
			BiometricType bioType = segment.getBdbInfo().getType().get(0);
			List<String> bioSubTypeList = segment.getBdbInfo().getSubtype();
			String bioSubType = "";
			if (bioSubTypeList != null && !bioSubTypeList.isEmpty())
				bioSubType = bioSubTypeList.get(0);

			String key = bioType + "_" + bioSubType;
			// ignore modalities that are not to be matched
			if (!isValidBiometricType(bioType, modalitiesToExtract))
				continue;

			if (!values.containsKey(key)) {
				values.put(key, Util.encodeToURLSafeBase64(segment.getBdb()));
			}
		}
		Map<String, String> responseValues = null;
		try {			
			responseValues = extractFace(values);
			List<BIR> birList = record.getSegments();
			for (int index = 0; index < birList.size(); index++) {
				BIR segment = birList.get(index);
				BiometricType bioType = segment.getBdbInfo().getType().get(0);
				List<String> bioSubTypeList = segment.getBdbInfo().getSubtype();
				String bioSubType = "";
				if (bioSubTypeList != null && !bioSubTypeList.isEmpty())
					bioSubType = bioSubTypeList.get(0);

				String key = bioType + "_" + bioSubType;
				// ignore modalities that are not to be matched
				if (!isValidBiometricType(bioType, modalitiesToExtract))
					continue;

				if (responseValues != null && responseValues.containsKey(key)) {
					segment.getBirInfo().setPayload(segment.getBdb());
					segment.getBdbInfo().setLevel(ProcessedLevelType.INTERMEDIATE);
					segment.setBdb(Util.decodeURLSafeBase64(responseValues.get(key)));
				}
				birList.set(index, segment);
			}
			record.setSegments(birList);
			response.setStatusCode(200);
			response.setResponse(record);
		} catch (ExtractorException ex) {
			LOGGER.error("extractTemplate -- error", ex);
			switch (ExtractorErrorCode.fromErrorCode(ex.getErrorCode())) {
			case SOURCE_NOT_VALID_FACE_ISO_FORMAT_EXCEPTION:
			case SOURCE_NOT_VALID_BASE64URLENCODED_EXCEPTION:
			case COULD_NOT_READ_ISO_IMAGE_DATA_EXCEPTION:
			case NO_FACE_FOUND_EXCEPTION:
			case MORE_THAN_ONE_FACE_FOUND_EXCEPTION:
			case RELU_FEATURE_NULL_EXCEPTION:
			case RELU_BIAS_NULL_EXCEPTION:
			case FC_FEATURE_NULL_EXCEPTION:
			case FC_WEIGHT_NULL_EXCEPTION:
			case FEATURE_NULL_EXCEPTION:
			case WEIGHT_NULL_EXCEPTION:
			case FEATURE_MATRIX_BOX_NULL_EXCEPTION:
			case IMAGE_MATRIX_EMPTY_OR_IMAGE_TYPE_WRONG_EXCEPTION:
			case IMAGE_MATRIX_BOUNDING_BOX_NULL_EXCEPTION:
			case SOFT_MAX_BOUNDING_BOX_NULL_EXCEPTION:
			case REFINE_NET_BOUNDING_BOX_NULL_OR_EMPTY_EXCEPTION:
			case MAX_POOLING_PROPOSE_BOX_NULL_EXCEPTION:
						
				response.setStatusCode(401);
				response.setResponse(null);
				break;

			case SOURCE_CAN_NOT_BE_EMPTY_OR_NULL_EXCEPTION:
				response.setStatusCode(404);
				response.setResponse(null);
				break;

			default:
				response.setStatusCode(500);
				response.setResponse(null);
				break;
			}
		} catch (Exception ex) {
			LOGGER.error("extractTemplate -- error", ex);
			response.setStatusCode(500);
			response.setResponse(null);
		}

		return response;
	}
	
	private Map<String, String> extractFace(Map<String, String> values)
	{
		ExtractorErrorCode errorCode = null;
		Map<String, String> targetValues = new HashMap<String, String> ();
		for (Map.Entry<String,String> entry : values.entrySet())
		{
			String isoData = entry.getValue();
			String targetValue = null;
			byte[] imageData = mtcnn.getFaceDataFromISO(isoData);
			Mat src = Imgcodecs.imdecode(new MatOfByte(imageData), Imgcodecs.IMREAD_GRAYSCALE);
			ArrayList<FaceInfo> faces = mtcnn.findFace(imageData);
			if (faces != null && faces.size() == 1)
			{
				targetValue = getExtractedFace (src, faces.get(0));
			}
			else if (faces != null && faces.size() > 1)
			{
				List<FaceRect> facesInfo = new ArrayList<FaceRect>();
				for (FaceInfo face: faces)
				{
					facesInfo.add(new FaceRect(face.getFaceRect().x, face.getFaceRect().y, face.getFaceRect().width, face.getFaceRect().height));
				}
				
				Collections.sort(facesInfo, new FaceRectComparator());
				Rect rect_crop = facesInfo.get(0);
				FaceInfo faceInfo = null;
				for (FaceInfo face: faces)
				{
					if (face.getFaceRect().x == rect_crop.x && 
						face.getFaceRect().y == rect_crop.y &&  
						face.getFaceRect().width == rect_crop.width &&
						face.getFaceRect().height == rect_crop.height){
						faceInfo = face;
						break;
					}
				}
				targetValue = getExtractedFace (src, faceInfo);
			}
			else
			{
				errorCode = ExtractorErrorCode.NO_FACE_FOUND_EXCEPTION;
				throw new ExtractorException (errorCode.getErrorCode(), errorCode.getErrorMessage());
			}
			targetValues.put(entry.getKey(), targetValue);
		}
		return targetValues;
	}
	
	private String getExtractedFace(Mat src, FaceInfo faceAnnotation) {
		Mat faceAligner = Util.faceAligner(src, faceAnnotation);

		Imgproc.resize(faceAligner, faceAligner, new Size(45,45));
		MatOfInt map = new MatOfInt(Imgcodecs.IMWRITE_WEBP_QUALITY, 30);
		
		MatOfByte mem = new MatOfByte();
		Imgcodecs.imencode(".webp", faceAligner, mem, map);
		Util.encodeToURLSafeBase64(mem.toArray());
		return null;
	}

	private boolean isValidBiometricType(BiometricType bioType, List<BiometricType> modalitiesToExtract) {
		for (BiometricType biometricType : modalitiesToExtract) {
			if (biometricType == BiometricType.FACE)
			{
				return true;
			}
		}
		return false;
	}

	@Override
	public Response<BiometricRecord> segment(BiometricRecord sample, List<BiometricType> modalitiesToSegment,
			Map<String, String> flags) {
		// TODO Auto-generated method stub
		return null;
	}

	@Override
	public Response<BiometricRecord> convertFormatV2(BiometricRecord sample, String sourceFormat, String targetFormat,
			Map<String, String> sourceParams, Map<String, String> targetParams,
			List<BiometricType> modalitiesToConvert) {
		// TODO Auto-generated method stub
		return null;
	}	

	@Override
	public BiometricRecord convertFormat(BiometricRecord sample, String sourceFormat, String targetFormat,
			Map<String, String> sourceParams, Map<String, String> targetParams,
			List<BiometricType> modalitiesToConvert) {
		// TODO Auto-generated method stub
		return null;
	}	
}