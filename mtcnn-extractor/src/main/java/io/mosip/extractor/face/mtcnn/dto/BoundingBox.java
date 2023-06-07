package io.mosip.extractor.face.mtcnn.dto;

import lombok.Getter;
import lombok.Setter;
import lombok.ToString;

@Getter
@Setter
@ToString
public class BoundingBox {
	private float score;
	private int x1;
	private int y1;
	private int x2;
	private int y2;
	private float area;
	private boolean exist;
	private float[] keyPoints = new float[10];
	private float[] regreCoordinates = new float[4];
}
