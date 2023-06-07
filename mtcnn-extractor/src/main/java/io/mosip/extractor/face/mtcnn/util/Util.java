package io.mosip.extractor.face.mtcnn.util;

import java.nio.charset.StandardCharsets;
import java.util.ArrayList;
import java.util.Base64;
import java.util.Base64.Encoder;

import org.opencv.core.Mat;
import org.opencv.core.Point;
import org.opencv.core.Scalar;
import org.opencv.core.Size;

import io.mosip.extractor.face.mtcnn.dto.FaceInfo;

public class Util {
	public static void gemmCpu(int selfChannel, int lastChannel, int K, float alpha, ArrayList<Float> weightPData, int lda,
			ArrayList<Float> matrixPData, int ldb, float beta, ArrayList<Float> outboxPData, int ldc) {
		if (beta != 1) {
			for (int i = 0; i < selfChannel; ++i) {
				for (int j = 0; j < lastChannel; ++j) {
					int idxCount = i * ldc + j;
					outboxPData.set(idxCount, outboxPData.get(idxCount) * beta);
				}
			}
		}
		for (int t = 0; t < selfChannel; ++t) {
			for (int j = 0; j < lastChannel; ++j) {
				float sum = 0;
				for (int k = 0; k < K; ++k) {
					sum += alpha * weightPData.get(t * lda + k) * matrixPData.get(j * ldb + k);
				}
				outboxPData.set(t * ldc + j, outboxPData.get(t * ldc + j) + sum);
			}
		}
	}

	public static void gemvCpu(int selfChannel, int lastChannel, float alpha, ArrayList<Float> weightPData, int lda, 
		ArrayList<Float> pboxPData, int ldb, float beta, ArrayList<Float> outboxPData, int ldc) {
		for (int i = 0; i < selfChannel; i++) {
			float sum = 0;
			for (int j = 0; j < lastChannel; j++) {
				sum += weightPData.get(i * lda + j) * pboxPData.get(j * ldb);
			}
			outboxPData.set(i, sum);
		}
	}
	
	public static Mat faceAligner(Mat image, FaceInfo faceAnnotation) {
        //double[] desiredLeftEye = new double[]{0.27, 0.27};
        double[] desiredLeftEye = new double[]{0.32, 0.32};
        int desiredFaceWidth = faceAnnotation.getFaceRect().width;
        int desiredFaceHeight = faceAnnotation.getFaceRect().height;

        ArrayList<Float> keyPoints = faceAnnotation.getKeyPoints();
        //deciding to choose left and right eye
        
        Point leftEye = new Point(keyPoints.get(0), keyPoints.get(0 + 5));
        Point rightEye = new Point(keyPoints.get(1), keyPoints.get(1 + 5));
        
        //compute the angle between the eye centroids
        int dY = (int) (rightEye.y - leftEye.y);
        int dX = (int) (rightEye.x - leftEye.x);
        double angle = Math.toDegrees(Math.atan2(dY, dX));

        //compute the desired right eye x-coordinate based on the
        //desired x-coordinate of the left eye
        double desiredRightEyeX = 1.0 - desiredLeftEye[0];

        //determine the scale of the new resulting image by taking
        //the ratio of the distance between eyes in the *current*
        //image to the ratio of distance between eyes in the
        //*desired* image
        double dist = Math.sqrt((Math.pow(dX, 2)) + (Math.pow(dY, 2)));
        double desiredDist = desiredRightEyeX - desiredLeftEye[0];
        desiredDist = desiredDist * desiredFaceWidth;
        double scale = desiredDist / dist;

        //compute center (x, y)-coordinates (i.e., the median point)
        //between the two eyes in the input image
        Point eyesCenter = new Point(Math.floorDiv((int) (leftEye.x + rightEye.x), 2),
                Math.floorDiv((int) (leftEye.y + rightEye.y), 2));
        
        System.out.println("angle ::  "+ angle  + "  scale>>" + scale + " facerect>>" + faceAnnotation.getFaceRect().toString());
        //grab the rotation matrix for rotating and scaling the face
        Mat m = org.opencv.imgproc.Imgproc.getRotationMatrix2D(eyesCenter, angle, scale);
        Mat dst = null;
        double tX = desiredFaceWidth * 0.5;
        double tY = desiredFaceHeight * desiredLeftEye[1];

        // Convert to double (much faster than a simple for loop)
        //int CV_64F = 6;
        //m.convertTo(dst, CV_64F, 1, 0);
        //double arrOut[][] = new double[dst.rows()][dst.cols()];
        //for(int i = 0 ;i < dst.rows(); ++i)
        //for(int j = 0; j < dst.cols(); ++j)
        //arrOut[i][j] = dst.get(i, j)[0];//.at<Float>(i, j);
                
        //DoubleIndexer indexer = m.createIndexer();
        double eyeCenterX = m.get(0, 2)[0] + (tX - eyesCenter.x);
        double eyeCenterY = m.get(1, 2)[0] + (tY - eyesCenter.y);
        m.put(0, 2, new double[] {eyeCenterX});
        m.put(1, 2, new double[] {eyeCenterY});
        //arrOut[0] [2] = eyeCenterX;
        //arrOut[1] [2] = eyeCenterY;
        //indexer.release();

        Mat output = new Mat();

        int BORDER_CONSTANT = 0;
        /** bicubic interpolation */
        int INTER_CUBIC          = 2;
        org.opencv.imgproc.Imgproc.warpAffine(image, output, m, new Size(desiredFaceWidth, desiredFaceHeight),
                INTER_CUBIC, BORDER_CONSTANT, new Scalar(0.0, 0.0, 0.0, 0.0));

        return output;
    }
	
	public static double trignometryForDistance(Point a, Point b) {
	    return Math.sqrt(((b.x - a.x) * (b.x - a.x)) +
	                     ((b.y - a.y) * (b.y - a.y)));
	}
	private static Encoder urlSafeEncoder;

	static {
		urlSafeEncoder = Base64.getUrlEncoder().withoutPadding();
	}
	
	public static String encodeToURLSafeBase64(byte[] data) {
		if (isNullEmpty(data)) {
			return null;
		}
		return urlSafeEncoder.encodeToString(data);
	}

	public static String encodeToURLSafeBase64(String data) {
		if (isNullEmpty(data)) {
			return null;
		}
		return urlSafeEncoder.encodeToString(data.getBytes(StandardCharsets.UTF_8));
	}

	public static byte[] decodeURLSafeBase64(String data) {
		if (isNullEmpty(data)) {
			return null;
		}
		return Base64.getUrlDecoder().decode(data);
	}

	public static boolean isNullEmpty(byte[] array) {
		return array == null || array.length == 0;
	}

	public static boolean isNullEmpty(String str) {
		return str == null || str.trim().length() == 0;
	}
}
