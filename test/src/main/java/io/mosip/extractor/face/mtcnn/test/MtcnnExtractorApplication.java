package io.mosip.extractor.face.mtcnn.test;

import java.io.File;
import java.io.FileInputStream;
import java.io.FileNotFoundException;
import java.io.FileOutputStream;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.stream.Collectors;
import java.util.stream.Stream;

import org.opencv.core.Mat;
import org.opencv.core.MatOfByte;
import org.opencv.core.MatOfInt;
import org.opencv.core.Point;
import org.opencv.core.Rect;
import org.opencv.core.Scalar;
import org.opencv.core.Size;
import org.opencv.highgui.HighGui;
import org.opencv.imgcodecs.Imgcodecs;
import org.opencv.imgproc.Imgproc;
import org.opencv.videoio.VideoCapture;

import io.mosip.extractor.face.mtcnn.Mtcnn;
import io.mosip.extractor.face.mtcnn.dto.FaceInfo;
import io.mosip.extractor.face.mtcnn.dto.FaceRect;
import io.mosip.extractor.face.mtcnn.dto.FaceRectComparator;
import io.mosip.extractor.face.mtcnn.util.Util;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

public class MtcnnExtractorApplication {
	static Logger LOGGER = LoggerFactory.getLogger(MtcnnExtractorApplication.class);

	//public static String dirName="D://Project//Mosip//Mtcnn_Java//Mtcnn_Java//blind1";
    static {
		// load OpenCV library
        nu.pattern.OpenCV.loadShared();
        System.loadLibrary(org.opencv.core.Core.NATIVE_LIBRARY_NAME);
    }
    
	public static void main(String[] args) {
		long startTime = System.currentTimeMillis();
		
		if (args != null && args.length >= 1) {
			// Argument 1 should contain
			// "io.mosip.mtcnnextractor.image.folder.path"
			String biometricFolderPath = args[0];
			LOGGER.info("main :: biometricFolderPath :: Argument [1] " + biometricFolderPath);
			if (biometricFolderPath.contains(ApplicationConstant.MOSIP_BIOMETRIC_FOLDER_PATH)) {
				biometricFolderPath = biometricFolderPath.split("=")[1];
			} 
			String dirName = null;
			try {
				dirName = new File(".").getCanonicalPath();
			} catch (IOException e1) {
				// TODO Auto-generated catch block
				e1.printStackTrace();
			}
			dirName = dirName + biometricFolderPath;
			
			String testType = "image";
			if (testType.equals("image")) {
				List<String> files = null;
				try {
					files = findFiles(Paths.get(dirName), "jp2");
				} catch (IOException e) {
					// TODO Auto-generated catch block
					e.printStackTrace();
				}
				Mat image = null;
				Mtcnn mtcnn = new Mtcnn();
				for(String fileName:files)
				{
					long startTimeOneFile = System.currentTimeMillis();
					System.out.println("File Name : "+ fileName);
					//byte[] sData = readAllBytes(fileName);
					byte[] imageData = readAllBytes(fileName);
					//System.out.println("File Name : "+ new String (imageData));
					
					// if files are FACE ISO
					//byte[] sData = mtcnn.getFaceDataFromISO(Util.encodeToURLSafeBase64(imageData));
					// if files are JP2000
					byte[] sData = imageData;
	
					Mat src = Imgcodecs.imdecode(new MatOfByte(sData), Imgcodecs.IMREAD_GRAYSCALE);
					ArrayList<FaceInfo> faces = mtcnn.findFace(sData);
					Scalar scalar = new Scalar(0, 255, 0);
					
					if (faces != null && faces.size() > 0)
					{
						if (faces.size() == 1)
						{
							Rect rect_crop = faces.get(0).getFaceRect();
							
							try {
								
								Mat faceAligner = Util.faceAligner(src, faces.get(0));
	
								//Mat image_row = new Mat(src, rect_crop);
								//System.out.println("faceAligner : before >> " + faceAligner.toString());
								Imgproc.resize(faceAligner, faceAligner, new Size(45,45));
								//System.out.println("faceAligner : after >> " + faceAligner.toString());
								MatOfInt map = new MatOfInt(Imgcodecs.IMWRITE_WEBP_QUALITY, 30);
								
								MatOfByte mem = new MatOfByte();
								Imgcodecs.imencode(".webp", faceAligner, mem, map);
								String[] fileNames = fileName.split("\\\\");
								String outFileName = fileNames[fileNames.length-1].replaceAll(".jp2", "");
								saveAllBytes(dirName, outFileName + ".webp", mem.toArray());
							}
							catch(Exception e) {
								e.printStackTrace();
								System.out.println("Could not convert the face to webp :" + e.getMessage());
							}
						}
						else if (faces.size() > 1)
						{
							List<FaceRect> facesInfo = new ArrayList<FaceRect>();
							int index = 0;
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
									face.getFaceRect().height == rect_crop.height)
									{
										faceInfo = face;
										break;
									}
							}
							
							//System.out.println("rect_crop : "+ rect_crop.toString());
							try {
								//Mat image_row = new Mat(src, rect_crop);
								Mat faceAligner = Util.faceAligner(src, faceInfo);
								Imgproc.resize(faceAligner, faceAligner, new Size(45,45));
								//Imgproc.resize(image_row, image_row, new Size(45,45));
								MatOfInt map = new MatOfInt(Imgcodecs.IMWRITE_WEBP_QUALITY, 30);
								
								MatOfByte mem = new MatOfByte();
								Imgcodecs.imencode(".webp", faceAligner, mem, map);
								String[] fileNames = fileName.split("\\\\");
								String outFileName = fileNames[fileNames.length-1].replaceAll(".jp2", "");
								saveAllBytes(dirName, outFileName + ".webp", mem.toArray());
							}
							catch(Exception e) {
								e.printStackTrace();
								System.out.println("Could not convert the face to webp :" + e.getMessage());
							}
						}
						
						else
						{
							System.out.println("more face found in image File Name : "+ fileName);
						}
					}
					else
					{
						System.out.println("No face found in image File Name : "+ fileName);
					}
					
					// draw and show
					/*
					for (int i = 0; i < faces.size(); i++) {
						Point p1 = new Point(faces.get(i).faceRect.x, faces.get(i).faceRect.y);
						Point p2 = new Point(faces.get(i).faceRect.x + faces.get(i).faceRect.width - 1,
								faces.get(i).faceRect.y + faces.get(i).faceRect.height - 1);
						Imgproc.rectangle(image, p1, p2, scalar, 2);
	
						for (int num = 0; num < 5; num++) {
							Point kp = new Point(faces.get(i).keyPoints.get(num), faces.get(i).keyPoints.get(num + 5));
							Imgproc.circle(image, kp, 3, scalar, -1);
						}
					}
					*/
					long endTimeOneFile = System.currentTimeMillis();
					//System.out.println(fileName + " >>> Time Taken : " + (endTimeOneFile - startTimeOneFile)  + " milliseconds");
				}
				long endTime = System.currentTimeMillis();
				System.out.println("Time Taken : " + (endTime - startTime)  + " milliseconds");
	
				//HighGui.imshow("test", image);
				//HighGui.waitKey(0);
			} else {
				// initial video capture
				VideoCapture cap = new VideoCapture(0);
				if (!cap.isOpened()) {
					System.out.println("can not open cam");
					return;
				}
				//int imgW = (int) cap.get(Videoio.CAP_PROP_FRAME_WIDTH);
				//int imgH = (int) cap.get(Videoio.CAP_PROP_FRAME_HEIGHT);
	
				Mtcnn mtcnn = new Mtcnn();
				Scalar scalar = new Scalar(0, 255, 0);
				Mat frame = new Mat();
	
				while (true) {
					cap.read(frame);
					if (frame.empty()) {
						System.out.println("can not read image");
						break;
					}
	
					ArrayList<FaceInfo> faces = mtcnn.findFace(frame);
	
					// draw and show
					for (int i = 0; i < faces.size(); i++) {
						Point p1 = new Point(faces.get(i).getFaceRect().x, faces.get(i).getFaceRect().y);
						Point p2 = new Point(faces.get(i).getFaceRect().x + faces.get(i).getFaceRect().width - 1,
								faces.get(i).getFaceRect().y + faces.get(i).getFaceRect().height - 1);
						Imgproc.rectangle(frame, p1, p2, scalar, 2);
	
						for (int num = 0; num < 5; num++) {
							Point kp = new Point(faces.get(i).getKeyPoints().get(num), faces.get(i).getKeyPoints().get(num + 5));
							Imgproc.circle(frame, kp, 3, scalar, -1);
						}
					}
	
					HighGui.imshow("test", frame);
					HighGui.waitKey(1);
	
				}
				// release
				cap.release();
				frame.release();
			}
		}
	}
	
	public static byte[] readAllBytes (String fileName)
	{
		File file = new File (fileName);
		return loadFile (file);
	}
	
	public static byte[] loadFile (File file)
	{
		FileInputStream fin = null;

		byte[] fileContent = null;
		try
		{
			fin = new FileInputStream (file);
			fileContent = new byte[(int) file.length ()];
			fin.read (fileContent);
		}
		catch (IOException ex)
		{
			ex.printStackTrace();
			System.out.println("loadFile :: Error in getting file :: "+ ex.getLocalizedMessage());
		}
		finally
		{
			try
			{
				if (fin != null)
					fin.close ();
			}
			catch (Exception ex)
			{
				System.out.println("loadFile :: Error in closing file :: "+ ex.getLocalizedMessage());
			}
		}
		return fileContent;
	}
	
	public static List<String> findFiles(Path path, String fileExtension)
	        throws IOException {

        if (!Files.isDirectory(path)) {
            throw new IllegalArgumentException("Path must be a directory!");
        }

        List<String> result;

        try (Stream<Path> walk = Files.walk(path)) {
            result = walk
                    .filter(p -> !Files.isDirectory(p))
                    // this is a path, not string,
                    // this only test if path end with a certain path
                    //.filter(p -> p.endsWith(fileExtension))
                    // convert path to string first
                    .map(p -> p.toString().toLowerCase())
                    .filter(f -> f.endsWith(fileExtension))
                    .collect(Collectors.toList());
        }

        return result;
    }
	
	public static void saveAllBytes (String dirName, String fileName, byte [] data) throws FileNotFoundException, IOException
	{
		File outputFile = new File(dirName, fileName);
		//System.out.println("outputFile Name : "+ outputFile);
		outputFile.getParentFile().mkdirs(); // Will create parent directories if not exists
		outputFile.createNewFile();
		try (FileOutputStream outputStream = new FileOutputStream(outputFile)) {
		    outputStream.write(data);
		}
	}		
}
