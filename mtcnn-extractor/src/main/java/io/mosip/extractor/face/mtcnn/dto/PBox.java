package io.mosip.extractor.face.mtcnn.dto;

import java.util.ArrayList;

import lombok.Getter;
import lombok.Setter;
import lombok.ToString;

@Getter
@Setter
@ToString
//proposing different regions boxes
public class PBox {
	private ArrayList<Float> pData;
	private int width;
	private int height;
	private int channel;
}
