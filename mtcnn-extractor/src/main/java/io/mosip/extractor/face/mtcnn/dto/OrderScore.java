package io.mosip.extractor.face.mtcnn.dto;

import lombok.Getter;
import lombok.Setter;
import lombok.ToString;

@Getter
@Setter
@ToString
public class OrderScore {
	private float score;
	private int oriOrder;
}

