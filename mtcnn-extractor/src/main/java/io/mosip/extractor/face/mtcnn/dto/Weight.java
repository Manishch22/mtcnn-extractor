package io.mosip.extractor.face.mtcnn.dto;

import java.util.ArrayList;

import lombok.Getter;
import lombok.Setter;
import lombok.ToString;

@Getter
@Setter
@ToString
public class Weight {
	private ArrayList<Float> pData;
	private ArrayList<Float> pBias;
	private int lastChannel;
	private int selfChannel;
	private int kernelSize;
	private int stride;
	private int pad;
}