package io.mosip.extractor.face.mtcnn.dto;

import java.util.Comparator;

public class FaceRectComparator implements Comparator<FaceRect> {
    public int compare(FaceRect s1, FaceRect s2) {
        return Integer.compare(s2.width, s1.width);
    }
}