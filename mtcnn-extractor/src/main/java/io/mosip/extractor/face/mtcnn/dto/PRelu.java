package io.mosip.extractor.face.mtcnn.dto;

import java.util.ArrayList;

import lombok.Getter;
import lombok.Setter;
import lombok.ToString;

@Getter
@Setter
@ToString
//Parametric Rectified Linear Unit
public class PRelu {
	private ArrayList<Float> pData;
	private int width;
}
