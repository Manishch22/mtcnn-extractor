package io.mosip.extractor.face.mtcnn.dto;

import java.util.ArrayList;

import org.opencv.core.Rect;

import lombok.Getter;
import lombok.Setter;
import lombok.ToString;

@Getter
@Setter
@ToString
public class FaceInfo {
	private Rect faceRect;
	private ArrayList<Float> keyPoints;
}
