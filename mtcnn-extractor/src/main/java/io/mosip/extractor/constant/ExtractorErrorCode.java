package io.mosip.extractor.constant;

/**
 * ExtractorErrorCode Enum for the services errors.
 * 
 * @author Janardhan B S
 * @since 1.0.0
 */
public enum ExtractorErrorCode {
	FILE_NOT_FOUND_EXCEPTION("MOS-EXT-001", "Input Filename could not be found"),
	NUMBER_FORMAT_EXCEPTION("MOS-EXT-002", "Invalid Number format"),
	IO_EXCEPTION("MOS-EXT-003", "Invalid Input/Output"),
	SOURCE_CAN_NOT_BE_EMPTY_OR_NULL_EXCEPTION("MOS-EXT-004", "Source value can not be empty or null"),
	SOURCE_NOT_VALID_BASE64URLENCODED_EXCEPTION("MOS-EXT-005", "Source not valid base64urlencoded"),
	COULD_NOT_READ_ISO_IMAGE_DATA_EXCEPTION("MOS-EXT-006", "Could not read Source ISO Image Data"),
	SOURCE_NOT_VALID_FACE_ISO_FORMAT_EXCEPTION("MOS-EXT-007", "Source not valid ISO ISO19794_5_2011"),
	NOT_VALID_FACE_MODALITY_EXCEPTION("MOS-EXT-007", "Not valid Face modality"),
	NO_FACE_FOUND_EXCEPTION("MOS-EXT-008", "No valid Face found"),
	MORE_THAN_ONE_FACE_FOUND_EXCEPTION("MOS-EXT-009", "Mpre than one Face found"),
	RELU_FEATURE_NULL_EXCEPTION("MOS-EXT-010", "Relu feature is null"),
	RELU_BIAS_NULL_EXCEPTION("MOS-EXT-011", "Relu bias is null"),
	FC_FEATURE_NULL_EXCEPTION("MOS-EXT-012", "Fc feature is null"),
	FC_WEIGHT_NULL_EXCEPTION("MOS-EXT-013", "Fc weight is null"),
	FEATURE_NULL_EXCEPTION("MOS-EXT-014", "Feature is null"),
	WEIGHT_NULL_EXCEPTION("MOS-EXT-015", "Weight is null"),
	FEATURE_MATRIX_BOX_NULL_EXCEPTION("MOS-EXT-016", "Feature matrix box null"),
	IMAGE_MATRIX_EMPTY_OR_IMAGE_TYPE_WRONG_EXCEPTION("MOS-EXT-017", "ImageMatrix Image's type is wrong!!Please set CV_8UC3"),
	IMAGE_MATRIX_BOUNDING_BOX_NULL_EXCEPTION("MOS-EXT-018", "ImageMatrix bounding box null or empty"),
	SOFT_MAX_BOUNDING_BOX_NULL_EXCEPTION("MOS-EXT-019", "Softmax bounding box null or empty"),
	REFINE_NET_BOUNDING_BOX_NULL_OR_EMPTY_EXCEPTION("MOS-EXT-020", "RefineNet bounding box null or empty"),
	MAX_POOLING_PROPOSE_BOX_NULL_EXCEPTION("MOS-EXT-021", "MaxPooling feature2Matrix pbox is NULL"),

	TECHNICAL_ERROR_EXCEPTION("MOS-EXT-500", "Technical Error");

	private final String errorCode;
	private final String errorMessage;

	private ExtractorErrorCode(final String errorCode, final String errorMessage) {
		this.errorCode = errorCode;
		this.errorMessage = errorMessage;
	}

	public String getErrorCode() {
		return errorCode;
	}

	public String getErrorMessage() {
		return errorMessage;
	}
	
	public static ExtractorErrorCode fromErrorCode(String errorCode) {
		 for (ExtractorErrorCode paramCode : ExtractorErrorCode.values()) {
	     	if (paramCode.getErrorCode().equalsIgnoreCase(errorCode)) {
	        	return paramCode;
	    	}
	    }
		return TECHNICAL_ERROR_EXCEPTION;
	}
}
