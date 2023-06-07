package io.mosip.extractor.face.mtcnn.network;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileNotFoundException;
import java.io.FileOutputStream;
import java.io.FileReader;
import java.io.IOException;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.io.OutputStream;
import java.net.URI;
import java.nio.file.FileSystemAlreadyExistsException;
import java.nio.file.FileSystems;
import java.nio.file.Files;
import java.nio.file.LinkOption;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.Collections;

import org.apache.commons.io.IOUtils;

import io.mosip.extractor.constant.ExtractorErrorCode;
import io.mosip.extractor.exception.ExtractorException;
import io.mosip.extractor.face.mtcnn.dto.PBox;
import lombok.Getter;
import lombok.Setter;
import lombok.ToString;

@Getter
@Setter
@ToString
public abstract class Network {
	private String fileName;
	public Network()
	{
		super();
	}
	public Network(String fileName)
	{
		super();
		this.setFileName(fileName);
	}
	
	protected abstract void init();
	
	protected void image2MatrixInit(PBox pBox, int channel, int width, int height) {
		pBox.setChannel (channel);
		pBox.setWidth (width);
		pBox.setHeight (height);

		long byteLength = pBox.getChannel() * pBox.getHeight() * pBox.getWidth();
		pBox.setPData(new ArrayList<Float>());
		for (int i = 0; i < byteLength; i++) {
			pBox.getPData().add(0.0F);
		}
	}
	
	protected String getFileNameFromPath(String fileName)
	{
		String str = null;
		
		URI uri = null;
		try {
			uri = this.getClass().getResource("/" + fileName).toURI();
		} catch (java.net.URISyntaxException e1) {
			// TODO Auto-generated catch block
			e1.printStackTrace();
		}
		Path path;
		java.nio.file.FileSystem fileSystem = null;
		
		if (uri != null && uri.getScheme().equals("jar")) {
			try {
			    try {
					fileSystem = FileSystems.newFileSystem(uri, Collections.emptyMap());
				} catch (IOException e) {
					// TODO Auto-generated catch block
					e.printStackTrace();
				}
			}catch (FileSystemAlreadyExistsException e) {
				fileSystem = FileSystems.getFileSystem(uri);
			}
		    path = fileSystem.getPath("/BOOT-INF/classes" + "/" + fileName);
		} else {
		    path = Paths.get(uri);
		    return path.toAbsolutePath().toString();
		}
		
	//	save cascade to /temp
		str = "/tmp/" + fileName;
		Path strFile = Paths.get(str);
		if (!Files.exists(strFile,LinkOption.NOFOLLOW_LINKS)) {
			byte[] buffer = null;
			try {
				buffer = java.nio.file.Files.readAllBytes(path);
			} catch (IOException e) {
				// TODO Auto-generated catch block
				e.printStackTrace();
			}
			File targetFile = new File("/tmp/" + fileName);
			OutputStream outStream = null;
			try {
				outStream = new FileOutputStream(targetFile);
			} catch (FileNotFoundException e) {
				// TODO Auto-generated catch block
				e.printStackTrace();
			}
			try {
				outStream.write(buffer);
				outStream.close();
			} catch (IOException e) {
				// TODO Auto-generated catch block
				e.printStackTrace();
			}		
		}
		
		return str;
	}
	
	protected void readNetworkFileData(long dataNumber[], ArrayList<ArrayList<Float>> pTeam) {
		ExtractorErrorCode errorCode = null;
		BufferedReader in = null;
		try {
			InputStream is = this.getClass().getClassLoader().getResourceAsStream(getFileName());
			in = new BufferedReader(new InputStreamReader(is, "UTF-8"));
			//in = new BufferedReader(new FileReader(getFileNameFromPath(getFileName())));
		} catch (Exception e) {
			errorCode = ExtractorErrorCode.FILE_NOT_FOUND_EXCEPTION;
			throw new ExtractorException (errorCode.getErrorCode(), e.getLocalizedMessage());
		}
		String line;
		if (in != null) {
			int i = 0;
			int count = 0;
			int pTeam_count = 0;
			try {
				while ((line = in.readLine()) != null) {
					if (i < dataNumber[count]) {
						String newLine = line.substring(1, line.length() - 1);
						float data = Float.parseFloat(newLine);
						pTeam.get(count).set(pTeam_count++, data);
					} else {
						count++;
						dataNumber[count] += dataNumber[count - 1];
						pTeam_count = 0;

						String newLine = line.substring(1, line.length() - 1);
						float data = Float.parseFloat(newLine);
						pTeam.get(count).set(pTeam_count++, data);
					}
					i++;
				}
			} catch (NumberFormatException e) {	
				errorCode = ExtractorErrorCode.NUMBER_FORMAT_EXCEPTION;
				throw new ExtractorException (errorCode.getErrorCode(), e.getLocalizedMessage());
			} catch (IOException e) {
				errorCode = ExtractorErrorCode.IO_EXCEPTION;
				throw new ExtractorException (errorCode.getErrorCode(), e.getLocalizedMessage());
			} finally {
				try {
					in.close();
				} catch (IOException e) {
					errorCode = ExtractorErrorCode.IO_EXCEPTION;
					throw new ExtractorException (errorCode.getErrorCode(), e.getLocalizedMessage());
				}
			}
		} else {
			errorCode = ExtractorErrorCode.FILE_NOT_FOUND_EXCEPTION;
			throw new ExtractorException (errorCode.getErrorCode(), errorCode.getErrorMessage());
		}
	}

}
