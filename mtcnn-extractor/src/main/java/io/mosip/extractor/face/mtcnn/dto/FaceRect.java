package io.mosip.extractor.face.mtcnn.dto;

import org.opencv.core.Rect;

public class FaceRect extends Rect implements Comparable<FaceRect>
{
	public FaceRect(int x, int y, int width, int height) {
       super(x, y, width, height);
    }
	@Override
	public int compareTo(FaceRect r) {
		if( this.width != r.width) 
			return width-r.width;
        else 
        	return height-r.height;	
   }		
}