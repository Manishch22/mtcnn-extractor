package io.mosip.extractor.constant;

/**
 * MtcnnConstant information.
 * 
 * @author Janardhan B S
 * @since 1.0.0
 */
public class MtcnnConstant {
	public static float PNET_THRESHOLD = 0.6F;
	public static float PNET_NMS_THRESHOLD = 0.5F;
	public static float ONET_THRESHOLD = 0.7F;
	public static float RNET_THRESHOLD = 0.7F;

	public static float NMS_THRESHOLD = 0.7F;

	public static String BOUNDINGBOX_MODEL_UNION = "UNION";
	public static String BOUNDINGBOX_MODEL_MINIMUM = "MIN";
	public static String BOUNDINGBOX_MODEL_MAXIMUM = "MAX";

	public static int KEY_POINTS = 10;
	public static int RGB_CHANNELS = 3;
	public static int PNET_CHANNELS = 4;

	public static int ONET_IMAGE_MATRIX_WIDTH = 48;
	public static int ONET_IMAGE_MATRIX_HEIGHT = 48;

	public static int RNET_IMAGE_MATRIX_WIDTH = 24;
	public static int RNET_IMAGE_MATRIX_HEIGHT = 24;
}
